# Problem 2: Enterprise Knowledge Base (RAG) Pipeline

A production-grade Retrieval-Augmented Generation (RAG) knowledge base built with **LangChain**, **ChromaDB**, and **FastAPI**, featuring automated document cleaning, PII sanitization, chunking with rich metadata, two-stage retrieval with reranking, grounded answer generation with source citations, and direct integration with the **Q1 Voice Agent**.

---

## 🏗️ Architecture

```
                    ┌────────────────────────────┐
                    │  PDF / Markdown / CSV / JSON │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                         Universal Loaders
                                  │
                                  ▼
                     Cleaning & PII Redaction
                                  │
                                  ▼
                    Recursive Text Chunking
                                  │
                                  ▼
                       Metadata Enrichment
                                  │
                                  ▼
                     Dense Embeddings Model
                                  │
                                  ▼
                     ChromaDB Vector Store
                                  │
                                  ▼
                    Top-K Candidate Retrieval
                                  │
                                  ▼
                     Lexical/Semantic Reranker
                                  │
                                  ▼
                     Grounded LLM Prompting
                                  │
                                  ▼
                     Answer + Source Citations
                                  │
                        ┌─────────┴─────────┐
                        ▼                   ▼
                  FastAPI Service     Q1 Voice Agent
                 (JSON / Swagger)   (Speech Response)
```

---

## 📁 Directory Structure

```
.
├── data/
│   ├── raw/                      # Raw incoming documents (PDF, MD, CSV, JSON)
│   │   └── health_policy.md      # Sample comprehensive health policy
│   └── processed/                # Preprocessed and sanitized cache
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic environment configuration
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── document_loader.py    # Universal multi-format document loader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Header/footer/whitespace normalizer & deduplicator
│   │   └── pii_masker.py         # Regex & pattern PII sanitization
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py         # HuggingFace / Gemini / OpenAI embeddings factory
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma.py             # ChromaDB indexer, chunker, & vector manager
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # 2-stage retriever with candidate reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── rag_chain.py          # Grounded generator, citation extractor & voice summarizer
│   └── api/
│       ├── __init__.py
│       ├── schemas.py            # Pydantic request/response models
│       └── main.py               # FastAPI application & ingestion CLI
├── tests/
│   ├── __init__.py
│   ├── retrieval_tests.json      # Golden retrieval test cases (positive + out-of-domain negative)
│   └── test_rag.py               # Automated pytest suite
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: set `GOOGLE_API_KEY` or `OPENAI_API_KEY` and set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai`)*

### 4. Ingest Sample Knowledge Base
```bash
python -m app.api.main --ingest
```

### 5. Run the FastAPI Server
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Running Automated Tests

Run the full evaluation and test suite:
```bash
pytest tests/test_rag.py -v
```

---

## 🎙️ Q1 Voice Agent Integration

The `/api/v1/query` endpoint returns a specialized `speech_response` field:

```json
{
  "question": "What is the deductible for in-network care?",
  "answer": "The annual individual in-network deductible is $500, and the family deductible is $1,000. (Source: health_policy.md)",
  "speech_response": "The annual individual in-network deductible is $500, and the family deductible is $1,000.",
  "is_available": true,
  "citations": [
    {
      "source": "health_policy.md",
      "chunk_id": "health_policy.md_chunk_0",
      "category": "general_policy",
      "score": 0.82
    }
  ]
}
```
The Q1 voice bot simply streams the `speech_response` string directly to the text-to-speech (TTS) engine.
