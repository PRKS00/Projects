import os
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
        if self.llm_provider == "gemini":
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

        q_lower = question.lower()
        relevant_snippets = []
        for doc in docs:
            for sentence in doc.page_content.split(". "):
                if any(w in sentence.lower() for w in q_lower.split() if len(w) > 3):
                    clean_s = sentence.strip()
                    if clean_s and clean_s not in relevant_snippets:
                        relevant_snippets.append(clean_s)

        if not relevant_snippets:
            return "The requested information is unavailable in the knowledge base."

        sources = list({d.metadata.get("source", "health_policy.md") for d in docs})
        answer_text = " ".join(relevant_snippets[:3])
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
            prompt = PromptTemplate.from_template(RAG_SYSTEM_PROMPT).format(
                context=context_str,
                question=question,
            )
            response = self._llm.invoke(prompt)
            answer_text = response.content.strip()
        else:
            answer_text = self._mock_grounded_generator(question, retrieved_docs)

        is_available = "unavailable in the knowledge base" not in answer_text.lower()

        # 4. Generate Voice Agent speech response (short, clean, spoken tone for Q1 bot)
        if is_available:
            speech_response = answer_text.split("(Source:")[0].strip()
            # Clean formatting for speech
            speech_response = speech_response.replace("**", "").replace("#", "").replace("\n", " ")
        else:
            speech_response = "I checked the policy documents, but that information is currently unavailable."

        return {
            "question": question,
            "answer": answer_text,
            "speech_response": speech_response,
            "is_available": is_available,
            "citations": citations,
            "retrieved_count": len(retrieved_docs),
        }
