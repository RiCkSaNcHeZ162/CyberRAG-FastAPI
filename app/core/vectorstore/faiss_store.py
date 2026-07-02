"""
FAISS Vector Store - High-performance local vector storage and retrieval.

Supports:
- Adding vectors with metadata
- Similarity search (cosine)
- Persistence to disk
- Multiple collections
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS-based vector store with metadata management.

    Uses FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity)
    for fast and exact nearest-neighbor search.
    """

    def __init__(self, collection_name: str = "default", dimension: int = 384):
        self.collection_name = collection_name
        self.dimension = dimension
        self._store_path = settings.VECTORSTORE_DIR / collection_name

        # FAISS index - using flat inner product (cosine sim on L2-normed vectors)
        self.index: Optional[faiss.IndexFlatIP] = None
        # Metadata storage (parallel to FAISS vectors)
        self.metadata: List[Dict[str, Any]] = []
        # Chunk texts (parallel to FAISS vectors)
        self.texts: List[str] = []
        # Document ID tracking
        self.doc_ids: List[str] = []

        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index from disk or create a new one."""
        index_path = self._store_path / "index.faiss"
        meta_path = self._store_path / "metadata.pkl"

        if index_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    stored = pickle.load(f)
                self.metadata = stored.get("metadata", [])
                self.texts = stored.get("texts", [])
                self.doc_ids = stored.get("doc_ids", [])
                logger.info(
                    f"Loaded FAISS index '{self.collection_name}' "
                    f"with {self.index.ntotal} vectors"
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Creating new.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a fresh FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.texts = []
        self.doc_ids = []
        logger.info(f"Created new FAISS index '{self.collection_name}' (dim={self.dimension})")

    def add(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
        doc_id: str = "",
    ) -> int:
        """
        Add vectors with their corresponding texts and metadata.

        Args:
            vectors: numpy array of shape (n, dimension).
            texts: List of text strings corresponding to vectors.
            metadata_list: List of metadata dicts.
            doc_id: Document identifier for tracking.

        Returns:
            Number of vectors added.
        """
        if len(vectors) == 0:
            return 0

        # Ensure float32 and correct shape
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        assert vectors.shape[1] == self.dimension, (
            f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}"
        )

        # Add to FAISS index
        self.index.add(vectors)

        # Store metadata and texts
        self.texts.extend(texts)
        self.metadata.extend(metadata_list)
        self.doc_ids.extend([doc_id] * len(texts))

        logger.info(
            f"Added {len(texts)} vectors to '{self.collection_name}'. "
            f"Total: {self.index.ntotal}"
        )

        return len(texts)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: Query embedding of shape (dimension,).
            top_k: Number of results to return.
            doc_id_filter: Optional filter by document ID.

        Returns:
            List of dicts: {text, metadata, score, index}.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Search with extra results if filtering
        search_k = top_k * 3 if doc_id_filter else top_k
        search_k = min(search_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            # Apply document filter if specified
            if doc_id_filter and self.doc_ids[idx] != doc_id_filter:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(score),
                "index": int(idx),
            })

            if len(results) >= top_k:
                break

        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all vectors associated with a document ID.

        Note: FAISS doesn't support efficient deletion, so we rebuild the index.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of vectors deleted.
        """
        indices_to_keep = [
            i for i, did in enumerate(self.doc_ids) if did != doc_id
        ]
        deleted_count = len(self.doc_ids) - len(indices_to_keep)

        if deleted_count == 0:
            return 0

        # Reconstruct vectors from FAISS
        if indices_to_keep:
            kept_vectors = np.array([
                self.index.reconstruct(i) for i in indices_to_keep
            ], dtype=np.float32)

            self.texts = [self.texts[i] for i in indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            self.doc_ids = [self.doc_ids[i] for i in indices_to_keep]

            self._create_new_index()
            self.index.add(kept_vectors)
        else:
            self._create_new_index()

        logger.info(f"Deleted {deleted_count} vectors for doc_id '{doc_id}'")
        return deleted_count

    def save(self) -> None:
        """Persist index and metadata to disk."""
        self._store_path.mkdir(parents=True, exist_ok=True)

        index_path = self._store_path / "index.faiss"
        meta_path = self._store_path / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "texts": self.texts,
                "doc_ids": self.doc_ids,
            }, f)

        logger.info(f"Saved FAISS index '{self.collection_name}' ({self.index.ntotal} vectors)")

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        unique_docs = set(self.doc_ids)
        return {
            "collection_name": self.collection_name,
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "unique_documents": len(unique_docs),
            "document_ids": list(unique_docs),
        }

    def clear(self) -> None:
        """Clear the entire index."""
        self._create_new_index()
        logger.info(f"Cleared FAISS index '{self.collection_name}'")
