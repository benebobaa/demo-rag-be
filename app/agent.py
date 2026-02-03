from typing import List, Dict, Any, AsyncIterator
import os
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from app.rag_engine import rag_engine
from app.models import AgentTrace, QueryResponse
from app.stream_utils import build_stream_events

class KnowledgeAgent:
    # Default system prompt to guide agent behavior and tool selection
    DEFAULT_SYSTEM_PROMPT = """You are a detailed and thorough knowledge assistant with access to TWO tools:

TOOL SELECTION RULES:
1. **SearchDocuments**: Use for finding facts, definitions, summaries, or content from documents.
   - Example: "What is X?", "Summarize Y", "Find information about Z"
   
2. **ExploreGraph**: Use for finding RELATIONSHIPS, CONNECTIONS, or DEPENDENCIES between entities.
   - Example: "What is related to X?", "What connects A and B?", "Show dependencies of Y"
   - Keywords that trigger this: "hubungan", "related", "connected", "depends on", "links to"

OPERATIONAL GUIDELINES:
1. **Goal-Oriented Thoroughness**: Gather full information, BUT stop searching immediately if you have sufficient information to answer the user's question confidently.
2. **Iterative Search**: ONLY if a search result is vague or incomplete should you formulate a new query.
3. **Graph First for Relations**: If the user asks about relationships ("hubungan"), ALWAYS use ExploreGraph FIRST.
4. **Synthesis**: Combine information from multiple search steps to provide a rich, comprehensive answer.
5. **Language**: Answer in the same language as the user's question."""

    def __init__(self):
        self.rag = rag_engine
        # Configurable via environment variables
        self.recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", "10"))
        custom_prompt = os.getenv("AGENT_SYSTEM_PROMPT", "").strip()
        self.system_prompt = custom_prompt if custom_prompt else self.DEFAULT_SYSTEM_PROMPT
        
    def get_tools(self):
        return [
            Tool(
                name="SearchDocuments",
                func=self._search_docs,
                description="Useful for finding specific information in the uploaded documents. Input should be a search query."
            ),
            Tool(
                name="ExploreGraph",
                func=self._explore_graph,
                description="Useful for finding related concepts in the knowledge graph. Input should be an exact entity name."
            )
        ]

    @traceable(name="SearchDocuments", run_type="tool")
    def _search_docs(self, query: str) -> str:
        docs = self.rag.query_vector(query)
        if not docs:
            return "No documents found."
        return "\n\n".join([d.page_content for d in docs])

    @traceable(name="ExploreGraph", run_type="tool")
    def _explore_graph(self, entity: str) -> str:
        edges = self.rag.get_graph_neighbors(entity)
        if not edges:
            return f"No known relations for '{entity}'."
        return "\n".join([f"{e.source} --[{e.relation}]--> {e.target}" for e in edges])

    @traceable(name="RAG Query", run_type="chain")
    async def run_query(self, query: str, chat_history: List[Dict[str, str]] = [], session_id: str = None) -> QueryResponse:
        if not self.rag.llm:
            return QueryResponse(answer="LLM not configured. Please set DEEPSEEK_API_KEY.", trace=[])

        tools = self.get_tools()
        
        # Create LangGraph agent with system prompt to limit tool calls
        graph = create_react_agent(self.rag.llm, tools, prompt=self.system_prompt)
        
        # Prepare messages
        from langchain_core.messages import AIMessage, HumanMessage
        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        # Invoke with LangSmith config for rich tracing
        inputs = {"messages": messages}
        config = RunnableConfig(
            tags=["rag-query"],
            metadata={
                "session_id": session_id or "unknown",
                "query": query[:100],
                "history_length": len(chat_history),
            },
            recursion_limit=self.recursion_limit,  # Limit agent steps to prevent excessive tool calls
        )
        result = await graph.ainvoke(inputs, config=config)
        
        # Parse result
        messages = result["messages"]
        final_answer = messages[-1].content
        
        traces = []
        for msg in messages:
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                     traces.append(AgentTrace(
                        step="Action",
                        thought=f"Invoking {tool_call['name']} with {tool_call['args']}",
                        tool=tool_call["name"],
                        observation="Waiting for output..."
                     ))
            elif msg.type == "tool":
                traces.append(AgentTrace(
                    step="Observation",
                    thought=f"Output from Tool {msg.name if hasattr(msg, 'name') else ''}",
                    observation=str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                ))
            
        return QueryResponse(
            answer=final_answer,
            trace=traces
        )

    async def stream_query(self, query: str, chat_history: List[Dict[str, str]] = [], session_id: str = None) -> AsyncIterator[Dict[str, Any]]:
        if not self.rag.llm:
            yield {
                "type": "error",
                "code": "llm_not_configured",
                "message": "LLM not configured. Please set DEEPSEEK_API_KEY.",
                "recoverable": False,
            }
            return

        tools = self.get_tools()
        graph = create_react_agent(self.rag.llm, tools, prompt=self.system_prompt)

        from langchain_core.messages import AIMessage, HumanMessage
        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=query))
        inputs = {"messages": messages}
        
        # LangSmith config for streaming with metadata
        config = RunnableConfig(
            tags=["rag-stream"],
            metadata={
                "session_id": session_id or "unknown",
                "query": query[:100],
                "history_length": len(chat_history),
            },
            recursion_limit=self.recursion_limit,  # Limit agent steps to prevent excessive tool calls
        )

        yield {
            "type": "status",
            "stage": "planning",
            "message": "Planning response...",
        }

        final_answer = None
        event_count = 0
        events_without_yield = 0
        HEARTBEAT_INTERVAL = 20  # Emit heartbeat every N events without real output
        STREAM_TIMEOUT = 120  # Maximum seconds for entire stream
        
        import asyncio
        try:
            async with asyncio.timeout(STREAM_TIMEOUT):
                async for event in graph.astream_events(inputs, version="v2", config=config):
                    event_count += 1
                    
                    stream_events, answer = build_stream_events(event)
                    if answer:
                        final_answer = answer
                    
                    if stream_events:
                        events_without_yield = 0
                        for stream_event in stream_events:
                            yield stream_event
                    else:
                        events_without_yield += 1
                        # Emit heartbeat to keep connection alive during long processing
                        if events_without_yield >= HEARTBEAT_INTERVAL:
                            events_without_yield = 0
                            yield {
                                "type": "status",
                                "stage": "processing",
                                "message": "Processing...",
                            }
        except asyncio.TimeoutError:
            if final_answer:
                # We have a partial answer, still return it
                yield {
                    "type": "answer",
                    "answer": final_answer,
                }
            else:
                yield {
                    "type": "error",
                    "code": "timeout",
                    "message": f"Request timed out after {STREAM_TIMEOUT} seconds.",
                    "recoverable": True,
                }
            return
        except Exception as e:
            if final_answer:
                yield {
                    "type": "answer",
                    "answer": final_answer,
                }
            else:
                yield {
                    "type": "error",
                    "code": "stream_error",
                    "message": str(e),
                    "recoverable": False,
                }
            return

        if not final_answer:
            final_answer = "No response generated."

        yield {
            "type": "answer",
            "answer": final_answer,
        }

# Singleton
agent_system = KnowledgeAgent()



