from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from app.config import settings
from app.vectorstore.chroma import ChromaStoreManager


class AdvancedRetriever:
    """
    Two-stage retrieval pipeline:
    1. Broad candidate retrieval (Top-K = 10) from ChromaDB.
    2. Metadata filtering and Reranking (Top-N = 3) to boost precision and evidence density.
    """

    def __init__(self, vectorstore_mgr: Optional[ChromaStoreManager] = None):
        self.vectorstore_mgr = vectorstore_mgr or ChromaStoreManager()

    def _rerank_candidates(
        self, query: str, candidate_docs: List[Tuple[Document, float]], top_n: int
    ) -> List[Tuple[Document, float]]:
        """
        Reranks retrieved candidate chunks based on lexical overlap, semantic alignment,
        and header matching heuristics.
        """
        query_terms = set(query.lower().split())
        reranked: List[Tuple[Document, float]] = []

        for doc, initial_score in candidate_docs:
            content_lower = doc.page_content.lower()
            # Calculate term match bonus
            matched_terms = sum(1 for term in query_terms if term in content_lower)
            term_ratio = matched_terms / max(len(query_terms), 1)

            # Combined score: 60% semantic similarity + 40% lexical alignment
            combined_score = (0.6 * initial_score) + (0.4 * term_ratio)

            reranked.append((doc, combined_score))

        # Sort descending by combined rerank score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]

    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        Executes broad candidate retrieval, applies metadata filters,
        and reranks candidates to return the top N documents.
        """
        k = top_k or settings.RETRIEVER_TOP_K
        n = top_n or settings.RERANKER_TOP_N

        # 1. Broad retrieval with scores
        try:
            results_with_scores = self.vectorstore_mgr.similarity_search_with_relevance_scores(
                query, k=k
            )
        except Exception:
            raw_docs = self.vectorstore_mgr.similarity_search(query, k=k)
            results_with_scores = [(d, 0.5) for d in raw_docs]

        # 2. Metadata filtering
        filtered_candidates = []
        for doc, score in results_with_scores:
            if filter_metadata:
                matches = all(
                    doc.metadata.get(key) == val for key, val in filter_metadata.items()
                )
                if not matches:
                    continue
            filtered_candidates.append((doc, score))

        if not filtered_candidates:
            return []

        # 3. Reranking
        reranked_results = self._rerank_candidates(query, filtered_candidates, top_n=n)

        # Attach final score to metadata for transparency and citations
        final_docs: List[Document] = []
        for doc, final_score in reranked_results:
            doc.metadata["retrieval_score"] = round(final_score, 4)
            final_docs.append(doc)

        return final_docs
