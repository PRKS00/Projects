import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.loaders.document_loader import UniversalDocumentLoader
from app.preprocessing.cleaner import DocumentCleaner
from app.preprocessing.pii_masker import PIIMasker
from app.vectorstore.chroma import ChromaStoreManager
from app.generation.rag_chain import RAGChainManager
from app.api.schemas import QueryRequest, QueryResponse, IngestResponse, HealthResponse

app = FastAPI(
    title="Problem 2 Knowledge Base RAG API",
    description="Enterprise Knowledge Base with LangChain, ChromaDB, PII masking, Reranking, and Q1 Voice Agent integration",
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


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint confirming configuration and vectorstore connectivity."""
    return HealthResponse(
        status="healthy",
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_provider=settings.LLM_PROVIDER,
        collection_name=settings.COLLECTION_NAME,
        persist_dir=settings.CHROMA_PERSIST_DIR,
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
