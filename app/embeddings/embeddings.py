import os
from typing import Optional, List
from langchain_core.embeddings import Embeddings
from app.config import settings


class DeterministicMockEmbeddings(Embeddings):
    """
    Lightweight deterministic fallback embeddings for testing environments
    when external models or weights are not yet downloaded.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Create a float vector of length dim normalized between -1.0 and 1.0
        vec = [(b / 128.0) - 1.0 for b in h]
        while len(vec) < self.dim:
            vec.extend(vec)
        return vec[:self.dim]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Factory function returning an instantiated LangChain Embeddings model.
    Supported: 'huggingface', 'gemini', 'openai', 'mock'
    """
    provider = (provider or settings.EMBEDDING_PROVIDER).lower()

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.HUGGINGFACE_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception:
            # Fallback if sentence-transformers is not yet installed or downloaded
            return DeterministicMockEmbeddings()

    elif provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
        except Exception:
            return DeterministicMockEmbeddings()

    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key
            )
        except Exception:
            return DeterministicMockEmbeddings()

    else:
        return DeterministicMockEmbeddings()
