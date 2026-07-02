"""
Recursive Chunker - Hierarchical text splitting with overlap.

Uses LangChain's RecursiveCharacterTextSplitter logic with
configurable separators, chunk sizes, and overlap.
"""

import logging

from app.config import settings
from app.core.chunking.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators.

    Separators (in order of priority):
    1. Double newline (paragraphs)
    2. Single newline
    3. Sentence-ending punctuation
    4. Space (word boundary)
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
        ]

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split text recursively with overlap.

        Args:
            text: Text to split.
            metadata: Optional metadata for each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)

        # Merge into target size with overlap
        final_chunks = self._merge_with_overlap(raw_chunks)

        result = []
        for i, content in enumerate(final_chunks):
            result.append(
                Chunk(
                    content=content.strip(),
                    metadata={**base_metadata, "strategy": "recursive"},
                    chunk_id=f"rec_{i}",
                )
            )

        logger.info(f"Recursive chunking produced {len(result)} chunks")
        return result

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            elif remaining_separators:
                result.extend(self._recursive_split(part, remaining_separators))
            else:
                # Force split by character count
                for i in range(0, len(part), self.chunk_size):
                    result.append(part[i : i + self.chunk_size])

        return result

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge small segments into chunk_size blocks with overlap."""
        if not segments:
            return []

        chunks = []
        current_chunk = ""

        for segment in segments:
            if not current_chunk:
                current_chunk = segment
            elif len(current_chunk) + len(segment) + 1 <= self.chunk_size:
                current_chunk += " " + segment
            else:
                chunks.append(current_chunk)
                # Apply overlap from the end of the previous chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + " " + segment
                else:
                    current_chunk = segment

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks
