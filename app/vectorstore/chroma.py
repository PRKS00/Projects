from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import settings
from app.embeddings import get_embeddings


class ChromaStoreManager:
    """
    Manages vector storage in ChromaDB, including chunking,
    indexing, persistence, and collection maintenance.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function=None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embeddings = embedding_function or get_embeddings()

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazily initializes or retrieves the Chroma vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vectorstore

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """
        Splits documents into smaller overlapping chunks while preserving and
        enriching metadata with chunk indices.
        """
        size = chunk_size or settings.CHUNK_SIZE
        overlap = chunk_overlap or settings.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        chunks = splitter.split_documents(documents)

        # Enrich chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('filename', 'doc')}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["char_count"] = len(chunk.page_content)

        return chunks

    def add_documents(self, documents: List[Document]) -> int:
        """
        Chunks and ingests documents into the vector store.
        Returns the number of indexed chunks.
        """
        chunks = self.chunk_documents(documents)
        if chunks:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Performs raw similarity search for a query."""
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 5):
        """Performs similarity search with relevance scores."""
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def reset_collection(self) -> None:
        """Resets/deletes the collection."""
        try:
            self.vectorstore.delete_collection()
            self._vectorstore = None
        except Exception:
            pass
