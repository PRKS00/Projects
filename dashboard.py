"""
DarwixAI — Streamlit Knowledge Base & Voice Agent Dashboard
Hugging Face Free Models Edition
"""

import os
import json
import time
import streamlit as st
from pathlib import Path

# Configure page settings
st.set_page_config(
    page_title="DarwixAI — Enterprise RAG & Voice Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark cyber-slate styling
st.markdown("""
<style>
    .main { background-color: #090d16; }
    .stApp { background-color: #090d16; color: #f1f5f9; }
    .metric-box {
        background: rgba(18, 26, 44, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .citation-card {
        background: rgba(0, 0, 0, 0.35);
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .voice-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Pipeline lazy imports
from app.config import settings
from app.loaders.document_loader import UniversalDocumentLoader
from app.preprocessing.cleaner import DocumentCleaner
from app.preprocessing.pii_masker import PIIMasker
from app.vectorstore.chroma import ChromaStoreManager
from app.generation.rag_chain import RAGChainManager


@st.cache_resource
def get_pipeline():
    chroma_mgr = ChromaStoreManager()
    rag_chain = RAGChainManager()
    return chroma_mgr, rag_chain


chroma_mgr, rag_chain = get_pipeline()

# Sidebar Telemetry & Actions
with st.sidebar:
    st.title("⚡ DarwixAI Studio")
    st.caption("Enterprise RAG & Q1 Voice Agent")
    
    st.markdown("---")
    st.markdown(f"**Embedding Model:** `{settings.HUGGINGFACE_EMBEDDING_MODEL.split('/')[-1]}`")
    st.markdown(f"**LLM Model:** `{settings.HUGGINGFACE_LLM_MODEL.split('/')[-1]}`")
    st.markdown(f"**Collection:** `{settings.COLLECTION_NAME}`")
    
    chunk_count = 0
    try:
        if chroma_mgr.vectorstore and hasattr(chroma_mgr.vectorstore, "_collection"):
            chunk_count = chroma_mgr.vectorstore._collection.count()
    except Exception:
        chunk_count = 0
    st.metric("Total Indexed Chunks", chunk_count)
    
    st.markdown("---")
    if st.button("🔄 Sync & Re-index Pipeline", use_container_width=True):
        with st.spinner("Ingesting and indexing knowledge base..."):
            loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
            raw_docs = loader.load_all()
            cleaner = DocumentCleaner(deduplicate=True)
            cleaned = cleaner.clean_documents(raw_docs)
            masker = PIIMasker()
            sanitized = masker.mask_documents(cleaned)
            count = chroma_mgr.add_documents(sanitized)
            st.success(f"Indexed {count} sanitized chunks!")
            st.rerun()

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎙️ Voice & RAG Studio",
    "📞 Web Calling & Transcripts",
    "📁 Document Repository",
    "🛡️ PII Masker Sandbox",
    "🧠 Vector Store Explorer",
    "🧪 Golden Benchmarks",
])

# ==========================================
# TAB 1: RAG & VOICE STUDIO
# ==========================================
with tab1:
    st.subheader("Interactive Knowledge Query & Voice Agent")
    
    col_input, col_tune = st.columns([3, 1])
    with col_tune:
        top_k = st.slider("Top-K Candidates", 3, 25, settings.RETRIEVER_TOP_K)
        top_n = st.slider("Reranked Top-N", 1, 10, settings.RERANKER_TOP_N)
    
    with col_input:
        prompt_choice = st.selectbox(
            "Quick Prompts or Custom Input:",
            [
                "Custom Prompt...",
                "What is the annual individual in-network deductible?",
                "Who is eligible for Health Plus insurance?",
                "What are the copays for prescription medications?",
                "Are cosmetic surgeries or aromatherapy covered?",
                "What is the contact info for Alice Johnson in claims?",
                "What is the cancellation refund policy for international flights?",
            ]
        )
        if prompt_choice != "Custom Prompt...":
            default_query = prompt_choice
        else:
            default_query = ""
            
        user_query = st.text_area("Ask a policy question:", value=default_query, placeholder="Enter your question...")
        submit_btn = st.button("🚀 Execute RAG Query", type="primary")

    if submit_btn and user_query:
        with st.spinner("Retrieving from Hugging Face index & generating answer..."):
            t0 = time.perf_counter()
            response = rag_chain.query(question=user_query, top_k=top_k, top_n=top_n)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.markdown("### Grounded Answer")
                if response["is_available"]:
                    st.success("✅ Grounded Context Available")
                else:
                    st.warning("⚠️ Information Refusal / Out-of-Domain")
                st.write(response["answer"])
                
                # Voice Agent Spoken Response
                st.markdown("### 🎙️ Q1 Voice Agent Spoken Speech")
                st.markdown(f"""
                <div class="voice-box">
                    <strong>Spoken Response:</strong><br>"{response['speech_response']}"
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"### Citations ({len(response['citations'])})")
                st.caption(f"Latency: {latency_ms} ms")
                for c in response["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>📄 {c.get('source', 'Doc')}</strong> (Score: {c.get('score', 0.0):.3f})<br>
                        <small>"{c.get('excerpt', '')}"</small>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: WEB CALLING & TRANSCRIPTS
# ==========================================
with tab2:
    st.subheader("📞 Web Phone & Call Center Hub")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 0.5rem 0; color: #34d399;">📞 Toll-Free Callable Gateway: +1 (800) 327-9492</h3>
        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">
            Direct Telephony & WebRTC Voice Interface with Hugging Face Grounded Inference, PII Sanitization, and Session Recording.
            <br><strong>SIP URI:</strong> <code>sip:agent@darwix.ai</code> | <strong>Webhooks:</strong> <code>/api/v1/voice/incoming-call</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎙️ Instant Simulated Voice Call")
    call_col1, call_col2 = st.columns([2, 1])
    
    with call_col1:
        test_call_sel = st.selectbox(
            "Select Call Scenario to Execute / Inspect:",
            [
                "CALL-2026-0819-01: Deductibles, Family Max, & Primary Care ($500 / $20)",
                "CALL-2026-0819-02: Prescription Tiers & Out-of-Network Coinsurance ($10 / 40%)",
                "CALL-2026-0819-03: Claims PII Redaction & Out-of-Domain Flight Refusal",
            ]
        )
        custom_caller_q = st.text_input("Or speak / ask custom phone question:", placeholder="e.g. What is the family deductible?")
        btn_start_sim_call = st.button("🚀 Start Web Call Session", type="primary")

    with call_col2:
        st.markdown("**Gateway Telemetry:**")
        st.caption("• Status: 🟢 Online (Twilio / WebRTC Ready)")
        st.caption("• Voice Model: Q1 Web Speech Engine")
        st.caption("• PII Redaction: Enforced")
        st.caption("• Storage: `data/call_records.json`")

    if btn_start_sim_call:
        query_to_run = custom_caller_q if custom_caller_q else (
            "What is the annual individual in-network deductible under Health Plus?" if "01" in test_call_sel else
            ("What is the co-payment for Tier 1 generic medications?" if "02" in test_call_sel else
             "What is the contact information for Alice Johnson in claims?")
        )
        with st.spinner("Connecting call & executing grounded RAG response..."):
            t0 = time.perf_counter()
            call_res = rag_chain.query(question=query_to_run)
            lat_ms = round((time.perf_counter() - t0) * 1000, 1)

            st.success(f"✅ Call Connected (Latency: {lat_ms} ms)")
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown(f"**👤 Caller Question:**")
                st.info(query_to_run)
            with c_right:
                st.markdown(f"**🤖 Q1 Voice Agent Spoken Response:**")
                st.markdown(f"""
                <div class="voice-box">
                    "{call_res['speech_response']}"
                </div>
                """, unsafe_allow_html=True)
                
            if call_res["citations"]:
                st.markdown(f"**📄 Verified Citations:**")
                for cit in call_res["citations"]:
                    st.caption(f"Source: `{cit.get('source')}` | Score: {cit.get('score', 0.0):.3f}")

    st.markdown("---")
    st.markdown("### 📋 Recorded Call Transcripts & Evaluation Audit")
    
    call_rec_file = settings.DATA_DIR / "call_records.json"
    if call_rec_file.exists():
        with open(call_rec_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        for r in records:
            with st.expander(f"📞 {r['call_id']} — {r['scenario']} ({r['duration_seconds']}s, {r['status'].upper()})"):
                meta_c1, meta_c2, meta_c3 = st.columns(3)
                meta_c1.markdown(f"**Caller:** `{r['caller_number']}`")
                meta_c2.markdown(f"**Avg Latency:** `{r['avg_latency_ms']} ms`")
                meta_c3.markdown(f"**Groundedness:** `{'✅ 100% Grounded' if r.get('is_grounded') else 'Domain Refusal'}`")
                
                st.markdown(f"**Summary:** {r['summary']}")
                
                st.markdown("#### Verbatim Transcript:")
                for turn in r.get("transcript", []):
                    speaker_label = "🤖 Agent" if turn['speaker'] == 'agent' else "👤 Caller"
                    st.markdown(f"`[{turn.get('timestamp_offset', '00:00')}]` **{speaker_label}:** {turn['text']}")
                    if turn.get("citations"):
                        for c in turn["citations"]:
                            st.caption(f"↳ *Citation:* `{c.get('source')}` — \"{c.get('excerpt')}\"")

# ==========================================
# TAB 3: DOCUMENT REPOSITORY
# ==========================================
with tab3:
    st.subheader("Raw Documents in Knowledge Base")
    uploaded_file = st.file_uploader("Upload New Document (.md, .txt, .pdf, .json, .csv)", type=["md", "txt", "pdf", "json", "csv"])
    if uploaded_file:
        settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = settings.DATA_RAW_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name}. Auto-indexing...")
        loader = UniversalDocumentLoader(settings.DATA_RAW_DIR)
        raw_docs = loader.load_all()
        cleaner = DocumentCleaner(deduplicate=True)
        cleaned = cleaner.clean_documents(raw_docs)
        masker = PIIMasker()
        sanitized = masker.mask_documents(cleaned)
        chroma_mgr.add_documents(sanitized)
        st.rerun()

    raw_files = list(settings.DATA_RAW_DIR.glob("*")) if settings.DATA_RAW_DIR.exists() else []
    for f in raw_files:
        if f.is_file() and not f.name.startswith("."):
            with st.expander(f"📄 {f.name} ({f.stat().st_size} bytes)"):
                with open(f, "r", encoding="utf-8", errors="ignore") as file_handle:
                    st.code(file_handle.read(), language="markdown")

# ==========================================
# TAB 4: PII MASKER SANDBOX
# ==========================================
with tab4:
    st.subheader("Live PII Sanitization & Normalization Sandbox")
    sample_text = st.text_area(
        "Enter text containing sensitive PII:",
        value="Patient: John Doe, SSN: 123-45-6789. Email: jdoe@clinic.org, Phone: (555) 019-2834, Card: 4532-8921-9034-1298."
    )
    if st.button("🛡️ Sanitize Text"):
        cleaner = DocumentCleaner()
        cleaned = cleaner.clean_text(sample_text)
        masker = PIIMasker()
        sanitized, counts = masker.mask_text(cleaned)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Text:**")
            st.code(sample_text)
        with c2:
            st.markdown(f"**Sanitized Text ({sum(counts.values())} Redactions):**")
            st.code(sanitized)
            st.json(counts)

# ==========================================
# TAB 5: VECTOR STORE EXPLORER
# ==========================================
with tab5:
    st.subheader("ChromaDB Vector Store Chunks")
    try:
        if chroma_mgr.vectorstore and hasattr(chroma_mgr.vectorstore, "_collection"):
            data = chroma_mgr.vectorstore._collection.get(limit=30)
            if data and "documents" in data and data["documents"]:
                for i, doc_text in enumerate(data["documents"]):
                    meta = data["metadatas"][i] if i < len(data["metadatas"]) else {}
                    cid = data["ids"][i]
                    with st.expander(f"Chunk {i+1}: {meta.get('chunk_id', cid)} ({meta.get('source', 'unknown')})"):
                        st.markdown(f"**Category:** `{meta.get('category', 'policy')}`")
                        st.write(doc_text)
            else:
                st.info("No chunks indexed yet.")
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")

# ==========================================
# TAB 6: GOLDEN BENCHMARKS
# ==========================================
with tab6:
    st.subheader("Golden Retrieval & Refusal Evaluation Suite")
    if st.button("🧪 Run Golden Tests"):
        bench_file = settings.BASE_DIR / "tests" / "retrieval_tests.json"
        if bench_file.exists():
            with open(bench_file, "r", encoding="utf-8") as f:
                cases = json.load(f)
            
            passed = 0
            results = []
            for case in cases:
                t0 = time.perf_counter()
                res = rag_chain.query(question=case["query"])
                lat = round((time.perf_counter() - t0) * 1000, 1)
                
                exp_avail = case.get("expected_available", True)
                is_avail = res.get("is_available", False)
                ans = res.get("answer", "")
                
                match = (exp_avail == is_avail)
                if match:
                    for t in case.get("expected_terms", []):
                        if t.lower() not in ans.lower():
                            match = False
                            break
                if match:
                    passed += 1
                
                results.append({
                    "Status": "✅ PASS" if match else "❌ FAIL",
                    "Query": case["query"],
                    "Latency (ms)": lat,
                    "Expected Avail": exp_avail,
                    "Actual Avail": is_avail,
                    "Answer Snippet": ans[:80] + "..." if len(ans) > 80 else ans
                })
            
            acc = round((passed / len(cases) * 100), 1)
            st.metric("Evaluation Accuracy", f"{acc}%", f"{passed}/{len(cases)} tests passed")
            st.dataframe(results, use_container_width=True)

