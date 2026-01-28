from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "concept"

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class UploadResponse(BaseModel):
    filename: str
    message: str
    graph: GraphData

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AgentTrace(BaseModel):
    step: str
    thought: str
    tool: Optional[str] = None
    observation: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    trace: List[AgentTrace]
    session_id: Optional[str] = None
    highlighted_path: Optional[GraphData] = None

# Search Tester Models
class SearchRequest(BaseModel):
    query: str
    k: int = 5  # Number of results, default 5

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    execution_time_ms: float

