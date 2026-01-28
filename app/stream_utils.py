from typing import Any, Dict, List, Optional, Tuple


def truncate_text(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def extract_final_answer(output: Any) -> Optional[str]:
    if not isinstance(output, dict):
        return None

    messages = output.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content")
        return str(content) if content is not None else None

    content = getattr(last, "content", None)
    return str(content) if content is not None else None


def extract_content_from_message(message: Any) -> Optional[str]:
    """Extract content from AIMessage, filtering out tool call responses."""
    if message is None:
        return None
    
    # Check if this is an AIMessage with tool_calls (not a final answer)
    tool_calls = None
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return None  # This is a tool call decision, not final answer
        content = message.get("content")
        return str(content) if content else None
    
    # LangChain AIMessage object
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return None  # This is a tool call decision, not final answer
    
    content = getattr(message, "content", None)
    if content:
        return str(content)
    return None


def build_stream_events(event: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not isinstance(event, dict):
        return [], None

    event_type = event.get("event")
    name = event.get("name")
    data = event.get("data") or {}
    events: List[Dict[str, Any]] = []
    answer: Optional[str] = None

    if event_type == "on_tool_start":
        tool_input = data.get("input")
        thought = f"Invoking {name} with {tool_input}"
        events.append({
            "type": "trace",
            "step": "Action",
            "thought": thought,
            "tool": name,
            "observation": "Running...",
        })
        events.append({
            "type": "status",
            "stage": "tool",
            "message": f"Running {name}...",
        })
    elif event_type == "on_tool_end":
        output = data.get("output")
        events.append({
            "type": "trace",
            "step": "Observation",
            "thought": f"Output from Tool {name}",
            "tool": name,
            "observation": truncate_text(str(output)),
        })
    elif event_type == "on_chat_model_end":
        # This captures the final AI response (only when NO tool_calls)
        output = data.get("output")
        if output:
            content = extract_content_from_message(output)
            if content:
                answer = content
                events.append({
                    "type": "status",
                    "stage": "answering",
                    "message": "Composing answer...",
                })
    elif event_type == "on_chain_end":
        # Fallback: check if final answer is in chain output
        output = data.get("output")
        extracted = extract_final_answer(output)
        if extracted:
            answer = extracted
            events.append({
                "type": "status",
                "stage": "answering",
                "message": "Composing answer...",
            })

    return events, answer


