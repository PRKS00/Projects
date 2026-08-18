import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.loaders.document_loader import UniversalDocumentLoader
from app.preprocessing.cleaner import DocumentCleaner
from app.preprocessing.pii_masker import PIIMasker
from app.vectorstore.chroma import ChromaStoreManager
from app.generation.rag_chain import RAGChainManager
from app.api.main import app


@pytest.fixture(scope="module")
def setup_knowledge_base():
    """Seeds the ChromaDB vectorstore with sanitized sample documents."""
    loader = UniversalDocumentLoader("data/raw")
    raw_docs = loader.load_all()
    assert len(raw_docs) > 0, "Raw documents should not be empty."

    cleaner = DocumentCleaner(deduplicate=True)
    cleaned = cleaner.clean_documents(raw_docs)

    masker = PIIMasker()
    sanitized = masker.mask_documents(cleaned)

    chroma_mgr = ChromaStoreManager()
    count = chroma_mgr.add_documents(sanitized)
    assert count > 0, "Indexed chunk count must be greater than 0."

    return chroma_mgr


def test_pii_masker_standalone():
    masker = PIIMasker()
    sample_text = "Contact Alice at alice@example.com or call 555-123-4567, SSN: 123-45-6789."
    masked, detected = masker.mask_text(sample_text)
    assert detected is True
    assert "[EMAIL_REDACTED]" in masked
    assert "[PHONE_REDACTED]" in masked
    assert "[SSN_REDACTED]" in masked
    assert "alice@example.com" not in masked


def test_document_cleaner_standalone():
    cleaner = DocumentCleaner()
    messy_text = "Header\n\n\n\nPage 1 of 10\n\nContent here.\n\n\n\n\nFooter"
    cleaned = cleaner.clean_text(messy_text)
    assert "Page 1 of 10" not in cleaned
    assert "\n\n\n" not in cleaned


def test_retrieval_and_citations_golden_set(setup_knowledge_base):
    rag_chain = RAGChainManager()
    tests_path = Path("tests/retrieval_tests.json")
    with open(tests_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    for case in test_cases:
        res = rag_chain.query(case["query"])
        
        # Check availability status
        assert res["is_available"] == case["expected_available"], (
            f"Case {case['id']} expected is_available={case['expected_available']}, got {res['is_available']}"
        )

        # Check expected terms in answer
        for term in case["expected_terms"]:
            assert term.lower() in res["answer"].lower(), (
                f"Case {case['id']}: term '{term}' not found in answer: '{res['answer']}'"
            )

        # Check citations for positive cases
        if case["expected_available"] and case["expected_source"]:
            assert len(res["citations"]) > 0
            sources = [c["filename"] or c["source"] for c in res["citations"]]
            assert any(case["expected_source"] in s for s in sources)


def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_fastapi_query_endpoint(setup_knowledge_base):
    client = TestClient(app)
    response = client.post(
        "/api/v1/query",
        json={"query": "What is the deductible for in-network care?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "$500" in data["answer"]
    assert "speech_response" in data
    assert len(data["citations"]) > 0
