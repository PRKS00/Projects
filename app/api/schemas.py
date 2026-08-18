from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to ask the knowledge base", example="Who is eligible for Health Plus?")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key-value filters (e.g. {'category': 'coverage'})")
    top_k: Optional[int] = Field(default=10, description="Initial broad retrieval candidate count")
    top_n: Optional[int] = Field(default=3, description="Reranked top results count")


class CitationItem(BaseModel):
    source: str
    filename: Optional[str] = None
    chunk_id: Optional[str] = None
    category: Optional[str] = None
    score: Optional[float] = None
    excerpt: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    speech_response: str = Field(..., description="Short conversational response formatted directly for Q1 Voice Bot audio synthesis")
    is_available: bool
    citations: List[CitationItem]
    retrieved_count: int


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    llm_provider: str
    collection_name: str
    persist_dir: str
