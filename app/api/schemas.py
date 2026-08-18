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
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None


class SanitizePreviewRequest(BaseModel):
    text: str = Field(..., description="Raw text containing potential PII or messy formatting")


class SanitizePreviewResponse(BaseModel):
    original: str
    cleaned: str
    sanitized: str
    masked_count: int


class DocumentItem(BaseModel):
    filename: str
    path: str
    size_bytes: int
    size_human: str
    format: str
    modified_time: str


class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentItem]


class ChunkItem(BaseModel):
    id: str
    chunk_id: str
    source: str
    category: str
    content: str
    char_count: int


class ChunkListResponse(BaseModel):
    count: int
    chunks: List[ChunkItem]


class BenchmarkResultItem(BaseModel):
    id: str
    query: str
    passed: bool
    is_available: bool
    expected_available: bool
    answer: str
    latency_ms: float
    retrieved_count: int
    citations: List[CitationItem]


class BenchmarkResponse(BaseModel):
    total_tests: int
    passed_tests: int
    accuracy_percent: float
    results: List[BenchmarkResultItem]


class StatsResponse(BaseModel):
    total_raw_documents: int
    total_indexed_chunks: int
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    top_n: int

