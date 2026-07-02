"""
Semantic Chunker - Splits text based on semantic similarity.

Instead of fixed-size chunks, this chunker detects topic shifts
by computing embedding similarity between consecutive sentences,
and splits at semantic boundaries.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    page_numbers: list[int] = field(default_factory=list)


class SemanticChunker:
    """
    Splits text into semantically coherent chunks by detecting topic boundaries.

    Algorithm:
    1. Split text into sentences
    2. Compute embeddings for each sentence
    3. Calculate cosine similarity between consecutive sentences
    4. Split at points where similarity drops below threshold
    5. Merge very small chunks with neighbors
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer | None = None,
        similarity_threshold: float = 0.45,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        if embedding_model is not None:
            self._model = embedding_model
        else:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple heuristics."""
        import re

        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Filter out empty strings and very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _compute_similarities(self, sentences: list[str]) -> np.ndarray:
        """Compute cosine similarity between consecutive sentence embeddings."""
        if len(sentences) < 2:
            return np.array([])

        embeddings = self._model.encode(sentences, show_progress_bar=False)
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(sim)

        return np.array(similarities)

    def _find_split_points(self, similarities: np.ndarray) -> list[int]:
        """Find indices where semantic similarity drops below threshold."""
        if len(similarities) == 0:
            return []

        # Use adaptive threshold: mean - std dev
        adaptive_threshold = max(
            self.similarity_threshold,
            float(np.mean(similarities) - np.std(similarities)),
        )

        split_points = []
        for i, sim in enumerate(similarities):
            if sim < adaptive_threshold:
                split_points.append(i + 1)  # Split after this sentence

        return split_points

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split text into semantic chunks.

        Args:
            text: The full text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects with semantic boundaries.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [
                Chunk(
                    content=text.strip(),
                    metadata={**base_metadata, "strategy": "semantic"},
                )
            ]

        similarities = self._compute_similarities(sentences)
        split_points = self._find_split_points(similarities)

        # Create chunks from split points
        chunks = []
        start = 0
        all_points = split_points + [len(sentences)]

        for end in all_points:
            chunk_text = " ".join(sentences[start:end]).strip()

            if chunk_text:
                # If chunk is too large, do a secondary split
                if len(chunk_text) > self.max_chunk_size:
                    sub_chunks = self._force_split(chunk_text)
                    for sc in sub_chunks:
                        chunks.append(
                            Chunk(
                                content=sc,
                                metadata={**base_metadata, "strategy": "semantic"},
                            )
                        )
                else:
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            metadata={**base_metadata, "strategy": "semantic"},
                        )
                    )
            start = end

        # Merge very small chunks
        chunks = self._merge_small_chunks(chunks)

        # Assign IDs
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"sem_{i}"

        logger.info(
            f"Semantic chunking produced {len(chunks)} chunks from {len(sentences)} sentences"
        )
        return chunks

    def _force_split(self, text: str) -> list[str]:
        """Force split large text at roughly max_chunk_size boundaries."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= self.max_chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are too small with their neighbors."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        buffer = chunks[0]

        for i in range(1, len(chunks)):
            if len(buffer.content) < self.min_chunk_size:
                buffer.content += " " + chunks[i].content
            else:
                merged.append(buffer)
                buffer = chunks[i]

        merged.append(buffer)
        return merged
