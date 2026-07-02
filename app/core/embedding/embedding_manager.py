"""
Embedding Manager - Manages HuggingFace embedding models.

Uses BAAI/bge-small-en-v1.5 (free, ~130MB) for generating
high-quality text embeddings locally.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Singleton manager for text embedding generation.

    Features:
    - Loads model once, reuses across requests
    - Batch embedding with progress tracking
    - Normalized embeddings for cosine similarity
    """

    _instance: Optional["EmbeddingManager"] = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([])

        # BGE models benefit from a query prefix
        if "bge" in settings.EMBEDDING_MODEL.lower():
            # For retrieval, prepend instruction
            processed_texts = texts
        else:
            processed_texts = texts

        embeddings = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string.

        For BGE models, prepends the retrieval instruction.

        Args:
            query: Query text.
            normalize: Whether to normalize.

        Returns:
            1D numpy array of shape (dimension,).
        """
        # BGE models use instruction prefix for queries
        if "bge" in settings.EMBEDDING_MODEL.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self._model.encode(
            [query],
            normalize_embeddings=normalize,
        )

        return np.array(embedding[0], dtype=np.float32)

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed document chunks (no query prefix).

        Args:
            documents: List of document text chunks.
            batch_size: Encoding batch size.
            show_progress: Show progress bar.

        Returns:
            numpy array of embeddings.
        """
        return self.embed_texts(
            documents,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self.dimension
