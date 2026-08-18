import json
import pytest
from pathlib import Path
from app.config import settings
from app.loaders.document_loader import UniversalDocumentLoader
from app.preprocessing.cleaner import DocumentCleaner
from app.preprocessing.pii_masker import PIIMasker
from app.vectorstore.chroma import ChromaStoreManager
from app.generation.rag_chain import RAGChainManager


@pytest.fixture(scope="module")
def prepared_chain():
    # Ingest document into test chroma store
    loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
    raw_docs = loader.load_all()
    cleaner = DocumentCleaner(deduplicate=True)
    cleaned = cleaner.clean_documents(raw_docs)
    masker = PIIMasker()
    sanitized = masker.mask_documents(cleaned)

    chroma_mgr = ChromaStoreManager()
    chroma_mgr.add_documents(sanitized)

    rag = RAGChainManager()
    return rag


def test_call_transcripts_integrity():
    """Verifies that call_records.json and test_call_transcripts.json are valid."""
    call_records_path = settings.DATA_DIR / "call_records.json"
    assert call_records_path.exists(), "call_records.json should exist"

    with open(call_records_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) >= 3, "At least three test calls should be recorded"
    for r in records:
        assert "call_id" in r
        assert "caller_number" in r
        assert "transcript" in r
        assert len(r["transcript"]) > 0
        assert r["status"] == "completed"


def test_call_1_deductibles_rag(prepared_chain):
    """Verifies RAG accuracy for Call 1 (Deductibles & Copays)."""
    res1 = prepared_chain.query("What is the annual individual in-network deductible under Health Plus?")
    assert res1["is_available"] is True
    assert "$500" in res1["answer"]

    res2 = prepared_chain.query("What is the co-pay for visiting a primary care physician in-network?")
    assert res2["is_available"] is True
    assert "$20" in res2["answer"]


def test_call_2_prescriptions_rag(prepared_chain):
    """Verifies RAG accuracy for Call 2 (Prescription Tiers & Out-of-network)."""
    res1 = prepared_chain.query("What is the co-payment for Tier 1 generic medications?")
    assert res1["is_available"] is True
    assert "$10" in res1["answer"]

    res2 = prepared_chain.query("What is the co-insurance for out-of-network non-emergency care?")
    assert res2["is_available"] is True
    assert "40%" in res2["answer"]


def test_call_3_pii_and_refusal_rag(prepared_chain):
    """Verifies RAG accuracy for Call 3 (PII Redaction & Refusal)."""
    res1 = prepared_chain.query("What is the contact information for Alice Johnson in claims?")
    assert res1["is_available"] is True
    assert "[EMAIL_REDACTED]" in res1["answer"]
    assert "[PHONE_REDACTED]" in res1["answer"]

    res2 = prepared_chain.query("What is the refund policy for cancelled international flight tickets?")
    assert res2["is_available"] is False
    assert "unavailable in the knowledge base" in res2["answer"].lower()
