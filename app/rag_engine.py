import os
from typing import Any, cast
import networkx as nx
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from app.models import GraphData, GraphNode, GraphEdge
from app.embedding_config import resolve_embedding_model, resolve_embedding_dimensions, resolve_embedding_provider
import spacy

# Load spaCy for basic entity fallback or preprocessing if needed
# nlp = spacy.load("en_core_web_sm")

class RAGEngine:
    def __init__(self):
        print(f"DEBUG: Initializing RAGEngine. DeepSeek Key Present: {bool(os.getenv('DEEPSEEK_API_KEY'))}")
        self.vector_store: Any | None = None
        self.graph = nx.Graph()
        self.documents = [] # List of dicts: {id, name, size, date, status}
        
        # Initialize embedding model based on provider
        provider = resolve_embedding_provider(os.environ)
        model_name = resolve_embedding_model(os.environ)
        dimensions = resolve_embedding_dimensions(os.environ)
        
        print(f"DEBUG: Embedding Provider: {provider}, Model: {model_name}, Dimensions: {dimensions}")
        
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            print("DEBUG: Using OpenAI Embeddings")
            kwargs = {"model": model_name}
            if dimensions:
                kwargs["dimensions"] = dimensions
            self.embeddings = OpenAIEmbeddings(**kwargs)
        elif provider == "google" and os.getenv("GOOGLE_API_KEY"):
            print("DEBUG: Using Google Embeddings")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                output_dimensionality=dimensions,
            )
        else:
            print(f"DEBUG: No valid embedding provider configured (provider={provider})")
            self.embeddings = None
            
        if os.getenv("DEEPSEEK_API_KEY"):
            api_key = os.getenv("DEEPSEEK_API_KEY")
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=cast(Any, api_key),
                base_url="https://api.deepseek.com",
                temperature=0
            )
        else:
            self.llm = None

        # Initialize Pinecone if configured
        if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_NAME"):
            print("DEBUG: Using Pinecone Vector Store")
            self.vector_store = PineconeVectorStore(
                index_name=os.getenv("PINECONE_INDEX_NAME"),
                embedding=self.embeddings
            )
        else:
            print("DEBUG: Using Ephemeral Chroma Vector Store")

    @traceable(name="Process File", run_type="chain")
    async def process_file(self, file_path: str, db_session=None) -> GraphData:
        # 1. Load Document
        docs = self._load_file(file_path)
        
        # 2. Split Text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # 3. Embed & Store in Vector DB
        if self.embeddings:
            if os.getenv("PINECONE_API_KEY") and self.vector_store is not None:
                # Append to existing index
                self.vector_store.add_documents(chunks)
            else:
                # Ephemeral new session
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    collection_name="temp_session"
                )
        
        # 4. Extract Graph Triples
        await self._build_graph(chunks, db_session)
        
        # 5. Track Document
        import datetime
        self.documents.append({
            "id": len(self.documents) + 1,
            "name": os.path.basename(file_path),
            "size": f"{os.path.getsize(file_path) / 1024:.1f} KB",
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "indexed"
        })
        
        return self.get_graph_data()

    def _load_file(self, path: str) -> List[Document]:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path)
        return loader.load()

    @traceable(name="Build Knowledge Graph", run_type="chain")
    async def _build_graph(self, chunks: List[Document], db_session=None):
        print("DEBUG: Starting _build_graph")
        if not self.llm:
            print("DEBUG: Self.llm is NONE. Skipping graph build.")
            return

        # Limit to first N chunks to save time/cost for demo
        demo_chunks = chunks[:5] 
        print(f"DEBUG: Processing {len(demo_chunks)} chunks for graph extraction.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledge graph extractor. Extract interesting entities and their relationships from the text. Return a JSON object with a key 'triples' which is a list of objects containing 'source', 'relation', 'target'."),
            ("user", "Text: {text}")
        ])
        
        chain = prompt | self.llm
        
        import asyncio
        # Run in parallel with semaphore if needed, but simple loop for now
        tasks = [chain.ainvoke({"text": c.page_content}) for c in demo_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"DEBUG: LLM returned {len(results)} results.")
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                print(f"DEBUG: Result {i} FAILED: {res}")
                continue
            
            # Helper to parse if string or AIMessage
            content = getattr(res, "content", res)
            
            parsed_res = None
            if isinstance(content, str):
                import json
                try:
                    # Clean code blocks if present
                    content = content.replace("```json", "").replace("```", "").strip()
                    parsed_res = json.loads(content)
                except Exception as e:
                    print(f"DEBUG: Failed to parse JSON: {e}")
                    continue
            elif isinstance(content, dict):
                parsed_res = content
            
            if parsed_res and "triples" in parsed_res:
                print(f"DEBUG: Saving {len(parsed_res['triples'])} triples from chunk {i}")
                for triple in parsed_res["triples"]:
                    s, r, t = triple.get("source"), triple.get("relation"), triple.get("target")
                    if s and r and t:
                        self.add_triple(s, r, t, db_session=db_session)
            else:
                print(f"DEBUG: No triples found in result {i}")

    def get_graph_data(self) -> GraphData:
        nodes = [GraphNode(id=n, label=data.get("label", n), type=data.get("type", "concept")) 
                 for n, data in self.graph.nodes(data=True)]
        edges = [GraphEdge(source=u, target=v, relation=data.get("relation", "related")) 
                 for u, v, data in self.graph.edges(data=True)]
        return GraphData(nodes=nodes, edges=edges)

    def get_documents(self) -> List[Dict[str, Any]]:
        return self.documents

    @traceable(name="Vector Search", run_type="retriever")
    def query_vector(self, query: str, k: int = 3) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    @traceable(name="Vector Search With Scores", run_type="retriever")
    def query_vector_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Query vector store and return documents with similarity scores.
        Returns list of (Document, score) tuples.
        Note: For Pinecone/Chroma, lower scores typically mean MORE similar.
        """
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
        
    def get_graph_neighbors(self, node_id: str) -> List[GraphEdge]:
        if node_id not in self.graph:
            return []
        edges = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            edges.append(GraphEdge(source=node_id, target=neighbor, relation=edge_data.get("relation", "related")))
        return edges

    def add_triple(self, source: str, relation: str, target: str, db_session=None):
        # 1. Update in-memory graph
        self.graph.add_node(source, label=source, type="concept")
        self.graph.add_node(target, label=target, type="concept")
        self.graph.add_edge(source, target, relation=relation)
        
        # 2. Persist if DB session provided
        if db_session:
            from app.models_db import GraphTriple
            # Check if exists (naive check for demo)
            exists = db_session.query(GraphTriple).filter_by(source=source, target=target, relation=relation).first()
            if not exists:
                new_triple = GraphTriple(source=source, relation=relation, target=target)
                db_session.add(new_triple)
                try:
                    db_session.commit()
                except:
                    db_session.rollback()

    def load_graph_from_db(self, db_session):
        print("DEBUG: Loading graph from DB...")
        from app.models_db import GraphTriple
        triples = db_session.query(GraphTriple).all()
        for t in triples:
            self.graph.add_node(t.source, label=t.source, type="concept")
            self.graph.add_node(t.target, label=t.target, type="concept")
            self.graph.add_edge(t.source, t.target, relation=t.relation)
        print(f"DEBUG: Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

# Singleton instance for demo simplicity
rag_engine = RAGEngine()
