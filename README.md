# DarwixAI: Enterprise Knowledge Base (RAG) & Q1 Voice Agent Pipeline

A production-grade, zero-cost Retrieval-Augmented Generation (RAG) knowledge base built with **LangChain**, **Hugging Face Free Models**, **ChromaDB**, and **FastAPI**, featuring automated document cleaning, PII sanitization, chunking with rich metadata, two-stage retrieval with reranking, grounded answer generation with source citations, a dark-mode interactive **Web Dashboard**, and direct integration with the **Q1 Voice Agent**.

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
                Hugging Face Dense Embeddings
               (sentence-transformers/all-MiniLM)
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
                   Hugging Face LLM Prompting
                      (google/flan-t5-base)
                                  │
                                  ▼
                     Answer + Source Citations
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
          FastAPI Backend   Interactive Web   Q1 Voice Agent
         (JSON / Swagger)      Dashboard      (TTS Synthesis)
```

---

## 🌟 Key Features

1. **🤗 Hugging Face Free Models (Zero API Cost)**:
   - **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Fast, dense 384-dim semantic representations).
   - **Generation:** `google/flan-t5-base` (Instruction-tuned, CPU-friendly) or free Hugging Face Inference endpoints.
2. **🎙️ Q1 Voice Agent Studio**:
   - Live speech synthesizer (Web Speech TTS) voicing formatted `speech_response` strings with animated audio equalizers.
   - Microphone Speech-to-Text (STT) input to speak queries directly into the knowledge base.
3. **📊 Cyber-Slate Interactive Web Dashboard**:
   - Modern dark-mode UI served directly at `http://localhost:8000/`.
   - Live RAG queries, prompt chips, citation cards, and retrieval context inspection.
   - Document upload & ingestion center with drag-and-drop support (`.md`, `.pdf`, `.txt`, `.csv`, `.json`).
   - Live interactive PII redaction sandbox and vector store chunk browser.
   - Browser-based golden benchmark evaluation runner (`tests/retrieval_tests.json`).
4. **🛡️ Automated Cleaning & PII Sanitization**:
   - Redacts SSNs, phone numbers, emails, and credit cards before vector indexing.
5. **🎯 Two-Stage Retrieval & Grounded Citations**:
   - Top-K candidate extraction + score reranking + strict source citation chips and out-of-domain refusal.

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
│   ├── config.py                 # Pydantic environment configuration (Hugging Face defaults)
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── document_loader.py    # Universal multi-format document loader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Header/footer/whitespace normalizer & deduplicator
│   │   └── pii_masker.py         # Regex & pattern PII sanitization
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py         # Hugging Face embeddings factory
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma.py             # ChromaDB indexer, chunker, & vector manager
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # 2-stage retriever with candidate reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   └── rag_chain.py          # Hugging Face grounded generator & Q1 voice formatter
│   ├── static/
│   │   ├── index.html            # Cyber-slate interactive web dashboard
│   │   ├── style.css             # Glassmorphic dark styling & audio wave animations
│   │   └── app.js                # Frontend controller & Web Speech TTS/STT orchestrator
│   └── api/
│       ├── __init__.py
│       ├── schemas.py            # Pydantic request/response models
│       └── main.py               # FastAPI application, static mounting & ingestion CLI
├── tests/
│   ├── __init__.py
│   ├── retrieval_tests.json      # Golden retrieval test cases (positive + out-of-domain negative)
│   └── test_rag.py               # Automated pytest suite
├── dashboard.py                  # Optional standalone Streamlit dashboard
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

### 3. Ingest Sample Knowledge Base
```bash
python -m app.api.main --ingest
```

### 4. Launch the Interactive Dashboard & API Server
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- 🌐 **Interactive Dashboard:** **[http://localhost:8000/](http://localhost:8000/)**
- 📖 **Swagger API Docs:** **[http://localhost:8000/docs](http://localhost:8000/docs)**

*(Optional) If you prefer Streamlit, you can also run:*
```bash
streamlit run dashboard.py
```

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
The Q1 voice bot streams the `speech_response` string directly to the text-to-speech engine or uses the built-in browser synthesizer in the Dashboard.
