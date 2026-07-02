"""
Hybrid Retriever - Combines vector search (FAISS) with keyword search (BM25).

Uses Reciprocal Rank Fusion (RRF) to merge results from both methods,
getting the best of semantic understanding AND exact keyword matching.
"""

import logging
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining dense (vector) and sparse (BM25) search.

    The alpha parameter controls the balance:
    - alpha = 1.0: Pure vector search
    - alpha = 0.0: Pure keyword search
    - alpha = 0.5: Equal weight (recommended)
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        embedding_manager: EmbeddingManager,
        alpha: float = settings.HYBRID_ALPHA,
    ):
        self.faiss_store = faiss_store
        self.embedding_manager = embedding_manager
        self.alpha = alpha

        # Build BM25 index from stored texts
        self._bm25: BM25Okapi | None = None
        self._bm25_texts: list[str] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from the texts in the FAISS store."""
        texts = self.faiss_store.texts
        if not texts:
            logger.info("No texts available for BM25 index")
            return

        # Tokenize for BM25
        tokenized = [self._tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_texts = texts
        logger.info(f"Built BM25 index with {len(texts)} documents")

    def rebuild_bm25(self) -> None:
        """Rebuild BM25 index (call after adding new documents)."""
        self._build_bm25_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        doc_id_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            doc_id_filter: Optional document ID filter.

        Returns:
            List of result dicts with 'text', 'metadata', 'score', 'retrieval_method'.
        """
        # 1. Vector search
        vector_results = self._vector_search(query, top_k * 2, doc_id_filter)

        # 2. BM25 keyword search
        bm25_results = self._bm25_search(query, top_k * 2, doc_id_filter)

        # 3. Merge with Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)

        logger.info(
            f"Hybrid retrieval: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(merged)} merged results"
        )

        return merged

    def _vector_search(
        self, query: str, top_k: int, doc_id_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform dense vector search via FAISS."""
        query_embedding = self.embedding_manager.embed_query(query)
        results = self.faiss_store.search(query_embedding, top_k, doc_id_filter)

        for r in results:
            r["retrieval_method"] = "vector"

        return results

    def _bm25_search(
        self, query: str, top_k: int, doc_id_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform sparse keyword search via BM25."""
        if self._bm25 is None or not self._bm25_texts:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            # Apply doc filter
            if doc_id_filter and self.faiss_store.doc_ids[idx] != doc_id_filter:
                continue

            results.append(
                {
                    "text": self._bm25_texts[idx],
                    "metadata": self.faiss_store.metadata[idx]
                    if idx < len(self.faiss_store.metadata)
                    else {},
                    "score": float(scores[idx]),
                    "index": int(idx),
                    "retrieval_method": "bm25",
                }
            )

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF Score = sum(1 / (k + rank)) where k=60 is a constant.

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            top_k: Final number of results.
            k: RRF constant (default 60).

        Returns:
            Merged and re-scored results.
        """
        # Score accumulator keyed by text content
        rrf_scores: dict[str, dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            text = result["text"]
            rrf_score = self.alpha * (1.0 / (k + rank + 1))

            if text not in rrf_scores:
                rrf_scores[text] = {
                    "text": text,
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "methods": [],
                    "vector_score": result.get("score", 0.0),
                    "bm25_score": 0.0,
                }
            rrf_scores[text]["rrf_score"] += rrf_score
            if "vector" not in rrf_scores[text]["methods"]:
                rrf_scores[text]["methods"].append("vector")

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            text = result["text"]
            rrf_score = (1 - self.alpha) * (1.0 / (k + rank + 1))

            if text not in rrf_scores:
                rrf_scores[text] = {
                    "text": text,
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "methods": [],
                    "vector_score": 0.0,
                    "bm25_score": result.get("score", 0.0),
                }
            rrf_scores[text]["rrf_score"] += rrf_score
            rrf_scores[text]["bm25_score"] = result.get("score", 0.0)
            if "bm25" not in rrf_scores[text]["methods"]:
                rrf_scores[text]["methods"].append("bm25")

        # Sort by RRF score and return top_k
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        final_results = []
        for item in sorted_results[:top_k]:
            final_results.append(
                {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": item["rrf_score"],
                    "retrieval_method": "+".join(item["methods"]),
                    "vector_score": item["vector_score"],
                    "bm25_score": item["bm25_score"],
                }
            )

        return final_results
