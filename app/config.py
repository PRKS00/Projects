import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")

    # Embeddings (Hugging Face Free Local Model)
    EMBEDDING_PROVIDER: str = "huggingface"  # "huggingface", "mock", "gemini", "openai"
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM (Hugging Face Free Model / Local Pipeline / Endpoint)
    LLM_PROVIDER: str = "huggingface"  # "huggingface", "huggingface_endpoint", "mock", "gemini", "openai"
    HUGGINGFACE_LLM_MODEL: str = "google/flan-t5-base"  # Fast, free CPU-friendly instruction model
    HUGGINGFACE_API_KEY: str = ""
    HF_TOKEN: str = ""
    
    # Optional Cloud API keys
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Vector DB
    COLLECTION_NAME: str = "knowledge_base"
    
    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Retrieval & Reranking
    RETRIEVER_TOP_K: int = 10
    RERANKER_TOP_N: int = 3
    SIMILARITY_THRESHOLD: float = 0.25

    # Server & Dashboard
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

