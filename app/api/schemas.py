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


# ==========================================
# Voice Calling & Telephony Schemas
# ==========================================

class PhoneConfigResponse(BaseModel):
    phone_number: str = Field(default="+1 (800) 327-9492", description="Callable Toll-Free Number (+1-800-DARWIX-AI)")
    phone_number_numeric: str = Field(default="+18003279492", description="E.164 Numeric format")
    sip_uri: str = Field(default="sip:agent@darwix.ai", description="Direct SIP Endpoint")
    webrtc_supported: bool = Field(default=True, description="Whether in-browser WebRTC / Web Audio calling is enabled")
    telephony_provider: str = Field(default="Twilio / Telnyx / SIP Trunk Ready", description="Telephony gateway compatibility")
    voice_agent_name: str = Field(default="Q1 Healthcare Voice Agent", description="Name of the conversational voice agent")
    webhook_inbound_url: str = Field(default="/api/v1/voice/incoming-call", description="Webhook for incoming carrier calls")
    webhook_gather_url: str = Field(default="/api/v1/voice/webhook/gather", description="Webhook for continuous telephony speech gather")


class CallTurn(BaseModel):
    turn_id: int
    speaker: str = Field(..., description="'caller' or 'agent'")
    text: str = Field(..., description="Spoken transcript of this turn")
    timestamp_offset: str = Field(default="00:00", description="Offset in MM:SS format")
    citations: Optional[List[CitationItem]] = Field(default=None, description="Citations if speaker is agent")
    latency_ms: Optional[float] = Field(default=None, description="Inference latency for this turn")


class CallRecord(BaseModel):
    call_id: str
    caller_number: str
    agent_number: str = "+1 (800) 327-9492"
    call_type: str = Field(default="Inbound Web Call", description="'Inbound Web Call', 'Telephony Call', or 'Simulated Test Call'")
    scenario: str
    timestamp: str
    duration_seconds: int
    status: str = "completed"
    overall_sentiment: str = "positive"
    is_grounded: bool = True
    avg_latency_ms: float
    summary: str
    transcript: List[CallTurn]
    rag_evaluations: Optional[List[Dict[str, Any]]] = None


class CallListResponse(BaseModel):
    count: int
    calls: List[CallRecord]


class CallTurnRequest(BaseModel):
    call_id: str = Field(..., description="Unique active call session identifier")
    user_speech: str = Field(..., description="Spoken utterance recognized from the caller")
    caller_number: Optional[str] = Field(default="+1 (555) 019-2834", description="Caller phone number or web client ID")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="List of previous turns [{'role': 'user'|'assistant', 'content': '...'}]")
    top_k: Optional[int] = Field(default=10)
    top_n: Optional[int] = Field(default=3)


class CallTurnResponse(BaseModel):
    call_id: str
    turn_id: int
    user_speech: str
    speech_response: str
    full_answer: str
    is_available: bool
    citations: List[CitationItem]
    latency_ms: float
    timestamp_offset: str


