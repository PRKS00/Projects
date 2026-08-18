import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.retrieval.retriever import AdvancedRetriever


RAG_SYSTEM_PROMPT = """You are a reliable business knowledge assistant.
Answer ONLY using the provided context below.
If the context does not contain enough information to directly answer the question, state clearly: "The requested information is unavailable in the knowledge base."
Never invent or extrapolate policy details.
Always cite the source document and section.

Context:
{context}

Question:
{question}

Helpful, Grounded Answer:"""


class RAGChainManager:
    """
    Manages the RAG pipeline end-to-end: query retrieval, prompt formatting,
    grounded LLM inference, source citation extraction, and voice agent speech formatting.
    """

    def __init__(
        self,
        retriever: Optional[AdvancedRetriever] = None,
        llm_provider: Optional[str] = None,
    ):
        self.retriever = retriever or AdvancedRetriever()
        self.llm_provider = (llm_provider or settings.LLM_PROVIDER).lower()
        self._llm = self._init_llm()

    def _init_llm(self):
        """Initializes the configured LLM provider or fallback."""
        if self.llm_provider in ("huggingface", "hf", "huggingface_local"):
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                from langchain_huggingface import HuggingFacePipeline

                model_id = settings.HUGGINGFACE_LLM_MODEL or "google/flan-t5-base"
                tok = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tok,
                    max_new_tokens=256,
                    temperature=0.1,
                    repetition_penalty=1.1,
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception:
                # Fallback gracefully to smart deterministic extractor if heavy weights not loaded
                return None

        elif self.llm_provider in ("huggingface_endpoint", "hf_endpoint"):
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                hf_token = (
                    settings.HUGGINGFACE_API_KEY
                    or settings.HF_TOKEN
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    or os.getenv("HF_TOKEN")
                )
                return HuggingFaceEndpoint(
                    repo_id=settings.HUGGINGFACE_LLM_MODEL or "HuggingFaceH4/zephyr-7b-beta",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.1,
                    max_new_tokens=256,
                )
            except Exception:
                return None

        elif self.llm_provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.0
                )
            except Exception:
                return None
        elif self.llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=api_key,
                    temperature=0.0
                )
            except Exception:
                return None
        return None

    def _format_context(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a numbered context block."""
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("chunk_id", f"chunk_{i}")
            formatted_chunks.append(f"[Source {i}: {source} | Ref: {section}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(formatted_chunks)

    def _extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extracts structured citation records from retrieved documents."""
        citations = []
        for doc in docs:
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "filename": doc.metadata.get("filename", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "category": doc.metadata.get("category", "policy"),
                "score": doc.metadata.get("retrieval_score", 0.0),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        return citations

    def _mock_grounded_generator(self, question: str, docs: List[Document]) -> str:
        """
        Deterministic grounded response generator when no external LLM API key is present.
        Ensures strict citation and refuses out-of-domain questions accurately.
        """
        if not docs:
            return "The requested information is unavailable in the knowledge base."

        stop_words = {
            "what", "which", "where", "when", "with", "from", "about", "policy",
            "document", "information", "does", "have", "tell", "need", "give",
            "know", "under", "help", "this", "that", "there", "these", "those",
            "will", "would", "could", "should", "your", "they", "them", "their",
            "the", "for", "and", "are", "per", "how", "much", "can", "you", "who", "all"
        }

        q_lower = question.lower()
        q_words = [w.strip("?,.:;!()\"'") for w in q_lower.split()]
        key_words = [w for w in q_words if len(w) > 2 and w not in stop_words]

        if not key_words:
            key_words = [w for w in q_words if len(w) > 2]

        relevant_snippets = []
        matched_any_keyword = False

        for doc in docs:
            # Paragraph level check
            paragraphs = doc.page_content.split("\n\n")
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean or p_clean.startswith("# Health Plus Premium"):
                    continue
                p_lower = p_clean.lower()
                
                # Check keyword matches
                matching_keys = [kw for kw in key_words if kw in p_lower]
                if matching_keys:
                    matched_any_keyword = True
                    # Remove section markdown headers from inline snippet text if present
                    clean_p = p_clean
                    for line in p_clean.split("\n"):
                        if line.startswith("## "):
                            clean_p = clean_p.replace(line, "").strip()
                    if clean_p and clean_p not in relevant_snippets:
                        relevant_snippets.append(clean_p)

        if not matched_any_keyword or not relevant_snippets:
            return "The requested information is unavailable in the knowledge base."

        # Extract clean source name (basename)
        sources = list({Path(d.metadata.get("source", "health_policy.md")).name for d in docs})
        answer_text = " ".join(relevant_snippets[:2])
        answer_text = answer_text.replace("\n- ", " ").replace("\n", " ").strip()
        if not answer_text.endswith("."):
            answer_text += "."

        return f"{answer_text} (Source: {', '.join(sources)})"


    def query(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end question answering pipeline with grounded citations.
        """
        # 1. Retrieve candidates
        retrieved_docs = self.retriever.retrieve(
            query=question,
            filter_metadata=filter_metadata,
            top_k=top_k,
            top_n=top_n,
        )

        citations = self._extract_citations(retrieved_docs)

        # 2. Check if relevant context exists
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "The requested information is unavailable in the knowledge base.",
                "speech_response": "I'm sorry, that information is not available in the policy documents.",
                "is_available": False,
                "citations": [],
                "retrieved_count": 0,
            }

        # 3. Generate Answer
        context_str = self._format_context(retrieved_docs)

        if self._llm:
            try:
                prompt = PromptTemplate.from_template(RAG_SYSTEM_PROMPT).format(
                    context=context_str,
                    question=question,
                )
                response = self._llm.invoke(prompt)
                if hasattr(response, "content"):
                    answer_text = response.content.strip()
                else:
                    answer_text = str(response).strip()
            except Exception:
                answer_text = self._mock_grounded_generator(question, retrieved_docs)
        else:
            answer_text = self._mock_grounded_generator(question, retrieved_docs)

        is_available = "unavailable in the knowledge base" not in answer_text.lower()

        # 4. Generate Voice Agent speech response (short, clean, spoken tone for Q1 bot)
        if is_available:
            speech_response = answer_text.split("(Source:")[0].strip()
            # Clean formatting for speech synthesis
            speech_response = speech_response.replace("**", "").replace("#", "").replace("\n", " ").strip()
            if not speech_response.endswith((".", "!", "?")):
                speech_response += "."
        else:
            speech_response = "I checked the policy documents, but that information is currently unavailable in the knowledge base."

        return {
            "question": question,
            "answer": answer_text,
            "speech_response": speech_response,
            "is_available": is_available,
            "citations": citations,
            "retrieved_count": len(retrieved_docs),
        }

