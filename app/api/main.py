import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from app.config import settings
from app.loaders.document_loader import UniversalDocumentLoader
from app.preprocessing.cleaner import DocumentCleaner
from app.preprocessing.pii_masker import PIIMasker
from app.vectorstore.chroma import ChromaStoreManager
from app.generation.rag_chain import RAGChainManager
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
    SanitizePreviewRequest,
    SanitizePreviewResponse,
    DocumentItem,
    DocumentListResponse,
    ChunkItem,
    ChunkListResponse,
    BenchmarkResultItem,
    BenchmarkResponse,
    StatsResponse,
)

app = FastAPI(
    title="DarwixAI Enterprise Knowledge Base RAG & Voice Agent API",
    description="Enterprise Knowledge Base with LangChain, ChromaDB, Hugging Face free models, PII masking, Reranking, and Q1 Voice Agent integration",
    version="1.0.0",
)

# Enable CORS for frontend / voice agent calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances
chroma_mgr = ChromaStoreManager()
rag_chain = RAGChainManager()

# Static assets directory
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def serve_dashboard():
    """Serves the interactive web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>DarwixAI Dashboard is initializing...</h1>")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint confirming configuration and vectorstore connectivity."""
    return HealthResponse(
        status="healthy",
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_provider=settings.LLM_PROVIDER,
        collection_name=settings.COLLECTION_NAME,
        persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_model=settings.HUGGINGFACE_EMBEDDING_MODEL,
        llm_model=settings.HUGGINGFACE_LLM_MODEL,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Analytics"])
def get_stats():
    """Retrieves operational telemetry and collection statistics."""
    raw_files = list(settings.DATA_RAW_DIR.glob("*")) if settings.DATA_RAW_DIR.exists() else []
    total_docs = len([f for f in raw_files if f.is_file() and not f.name.startswith(".")])

    chunk_count = 0
    try:
        if chroma_mgr.vectorstore and hasattr(chroma_mgr.vectorstore, "_collection"):
            chunk_count = chroma_mgr.vectorstore._collection.count()
    except Exception:
        chunk_count = 0

    return StatsResponse(
        total_raw_documents=total_docs,
        total_indexed_chunks=chunk_count,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        embedding_model=settings.HUGGINGFACE_EMBEDDING_MODEL,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.HUGGINGFACE_LLM_MODEL,
        collection_name=settings.COLLECTION_NAME,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.RETRIEVER_TOP_K,
        top_n=settings.RERANKER_TOP_N,
    )


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents():
    """
    Ingests all documents from `data/raw/`, applies cleaning, masks PII,
    splits into chunks with metadata, and indexes into ChromaDB.
    """
    try:
        # Step 1: Load documents
        loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
        raw_docs = loader.load_all()

        if not raw_docs:
            return IngestResponse(
                status="warning",
                files_processed=0,
                chunks_indexed=0,
                message="No documents found in data/raw/",
            )

        # Step 2: Clean documents
        cleaner = DocumentCleaner(deduplicate=True)
        cleaned_docs = cleaner.clean_documents(raw_docs)

        # Step 3: Mask PII
        masker = PIIMasker()
        sanitized_docs = masker.mask_documents(cleaned_docs)

        # Step 4: Chunk & Index into ChromaDB
        chunks_count = chroma_mgr.add_documents(sanitized_docs)

        return IngestResponse(
            status="success",
            files_processed=len(raw_docs),
            chunks_indexed=chunks_count,
            message=f"Successfully processed {len(raw_docs)} files into {chunks_count} sanitized chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with top-k retrieval, reranking, source citation,
    and voice-ready speech response for the Q1 Voice Bot.
    """
    try:
        result = rag_chain.query(
            question=request.query,
            filter_metadata=request.metadata_filter,
            top_k=request.top_k,
            top_n=request.top_n,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """Lists all files in the raw documents directory."""
    raw_dir = settings.DATA_RAW_DIR
    if not raw_dir.exists():
        return DocumentListResponse(count=0, documents=[])

    docs = []
    for p in sorted(raw_dir.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            size = p.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"

            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            docs.append(DocumentItem(
                filename=p.name,
                path=str(p),
                size_bytes=size,
                size_human=size_str,
                format=p.suffix.lstrip(".").upper() or "TXT",
                modified_time=mtime,
            ))

    return DocumentListResponse(count=len(docs), documents=docs)


@app.get("/api/v1/document-content", tags=["Documents"])
def get_document_content(filename: str = Query(...)):
    """Fetches text content of a specific raw document safely."""
    safe_path = settings.DATA_RAW_DIR / Path(filename).name
    if not safe_path.exists() or not safe_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
            return {"filename": filename, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.post("/api/v1/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...), auto_ingest: bool = Query(True)):
    """Uploads a new document to `data/raw/` and optionally indexes it immediately."""
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = settings.DATA_RAW_DIR / file.filename

    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)

    ingest_msg = ""
    chunks_added = 0
    if auto_ingest:
        try:
            loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
            raw_docs = loader.load_all()
            cleaner = DocumentCleaner(deduplicate=True)
            cleaned = cleaner.clean_documents(raw_docs)
            masker = PIIMasker()
            sanitized = masker.mask_documents(cleaned)
            chunks_added = chroma_mgr.add_documents(sanitized)
            ingest_msg = f" (Auto-indexed {chunks_added} chunks into knowledge base)"
        except Exception as e:
            ingest_msg = f" (Ingestion warning: {str(e)})"

    return {
        "status": "success",
        "filename": file.filename,
        "size_bytes": len(content),
        "message": f"Successfully uploaded {file.filename}{ingest_msg}",
    }


@app.post("/api/v1/sanitize-preview", response_model=SanitizePreviewResponse, tags=["Preprocessing"])
def preview_sanitization(request: SanitizePreviewRequest):
    """Interactive playground endpoint testing text normalization and PII redaction live."""
    cleaner = DocumentCleaner()
    cleaned = cleaner.clean_text(request.text)

    masker = PIIMasker()
    sanitized, counts = masker.mask_text(cleaned)
    total_masked = sum(counts.values())

    return SanitizePreviewResponse(
        original=request.text,
        cleaned=cleaned,
        sanitized=sanitized,
        masked_count=total_masked,
    )


@app.get("/api/v1/chunks", response_model=ChunkListResponse, tags=["Vector Store"])
def list_chunks(limit: int = Query(25, ge=1, le=100)):
    """Inspects stored chunks in the ChromaDB vector database."""
    try:
        if not chroma_mgr.vectorstore or not hasattr(chroma_mgr.vectorstore, "_collection"):
            return ChunkListResponse(count=0, chunks=[])

        data = chroma_mgr.vectorstore._collection.get(limit=limit)
        items = []
        if data and "ids" in data and data["ids"]:
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            for i in range(len(ids)):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                content = docs[i] if i < len(docs) and docs[i] else ""
                items.append(ChunkItem(
                    id=ids[i],
                    chunk_id=str(meta.get("chunk_id", ids[i])),
                    source=str(meta.get("source", "unknown")),
                    category=str(meta.get("category", "policy")),
                    content=content,
                    char_count=len(content),
                ))

        return ChunkListResponse(count=len(items), chunks=items)
    except Exception as e:
        return ChunkListResponse(count=0, chunks=[])


@app.get("/api/v1/benchmarks", response_model=BenchmarkResponse, tags=["Evaluation"])
def run_benchmarks():
    """Executes the golden retrieval test suite and calculates accuracy & latency."""
    benchmark_file = settings.BASE_DIR / "tests" / "retrieval_tests.json"
    if not benchmark_file.exists():
        raise HTTPException(status_code=404, detail="retrieval_tests.json not found.")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    passed_count = 0

    for case in cases:
        t0 = time.perf_counter()
        res = rag_chain.query(question=case["query"], top_k=settings.RETRIEVER_TOP_K, top_n=settings.RERANKER_TOP_N)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Check pass condition
        expected_avail = case.get("expected_available", True)
        is_avail = res.get("is_available", False)
        answer = res.get("answer", "")

        avail_match = (expected_avail == is_avail)
        terms_match = True
        for term in case.get("expected_terms", []):
            if term.lower() not in answer.lower():
                terms_match = False
                break

        passed = avail_match and terms_match
        if passed:
            passed_count += 1

        citations = res.get("citations", [])
        results.append(BenchmarkResultItem(
            id=case.get("id", "test"),
            query=case.get("query", ""),
            passed=passed,
            is_available=is_avail,
            expected_available=expected_avail,
            answer=answer,
            latency_ms=latency_ms,
            retrieved_count=res.get("retrieved_count", 0),
            citations=citations,
        ))

    accuracy = round((passed_count / len(cases) * 100), 1) if cases else 0.0
    return BenchmarkResponse(
        total_tests=len(cases),
        passed_tests=passed_count,
        accuracy_percent=accuracy,
        results=results,
    )


def run_ingest_cli():
    print("[*] Running document ingestion pipeline...")
    loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
    raw_docs = loader.load_all()
    print(f"[*] Loaded {len(raw_docs)} raw document(s).")

    cleaner = DocumentCleaner(deduplicate=True)
    cleaned = cleaner.clean_documents(raw_docs)
    print(f"[*] Cleaned {len(cleaned)} document(s).")

    masker = PIIMasker()
    sanitized = masker.mask_documents(cleaned)
    print(f"[*] Sanitized & masked PII in {len(sanitized)} document(s).")

    count = chroma_mgr.add_documents(sanitized)
    print(f"[+] Successfully indexed {count} chunks into ChromaDB at {settings.CHROMA_PERSIST_DIR}.")


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        run_ingest_cli()
    else:
        uvicorn.run("app.api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
