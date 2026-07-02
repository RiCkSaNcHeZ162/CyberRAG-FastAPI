"""
Context-Aware Chunker - Preserves document structure and metadata.

Extracts headers, page numbers, and section boundaries from PDF
content and attaches structure metadata to each chunk.
"""

import logging
import re

from app.config import settings
from app.core.chunking.recursive_chunker import RecursiveChunker
from app.core.chunking.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


class ContextAwareChunker:
    """
    Splits documents while preserving structural context.

    Features:
    - Detects headers and section titles
    - Preserves page number metadata
    - Attaches parent section context to each chunk
    - Creates breadcrumb-style metadata (Section > Subsection > ...)
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._recursive_chunker = RecursiveChunker(chunk_size, chunk_overlap)

        # Patterns for detecting headers / section titles
        self._header_patterns = [
            r'^#{1,6}\s+(.+)',                          # Markdown headers
            r'^(?:Chapter|Section|Part)\s+\d+[.:]\s*(.*)',  # Chapter/Section headings
            r'^(\d+\.)+\s+(.+)',                        # Numbered sections (1.2.3 Title)
            r'^[A-Z][A-Z\s]{3,}$',                    # ALL CAPS lines (likely headers)
            r'^(?:Abstract|Introduction|Conclusion|Summary|References|Appendix)\s*$',  # Standard sections
        ]

    def chunk_pages(
        self,
        pages: list[dict[str, any]],
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Chunk a list of page dictionaries with structure awareness.

        Args:
            pages: List of dicts with 'text' and 'page_number' keys.
            metadata: Additional metadata to attach.

        Returns:
            List of Chunk objects with structural context.
        """
        if not pages:
            return []

        base_metadata = metadata or {}
        all_chunks = []
        current_section = "Document Start"
        section_hierarchy = []

        for page_info in pages:
            text = page_info.get("text", "")
            page_num = page_info.get("page_number", 0)

            if not text.strip():
                continue

            # Detect sections within the page
            sections = self._extract_sections(text)

            for section_title, section_text in sections:
                if section_title:
                    current_section = section_title
                    section_hierarchy = self._update_hierarchy(
                        section_hierarchy, section_title
                    )

                # Chunk the section text
                section_chunks = self._recursive_chunker.chunk(section_text)

                for chunk in section_chunks:
                    chunk.metadata.update({
                        **base_metadata,
                        "strategy": "context_aware",
                        "page_number": page_num,
                        "section": current_section,
                        "section_hierarchy": " > ".join(section_hierarchy) if section_hierarchy else current_section,
                        "context_prefix": f"[Page {page_num} | {current_section}] ",
                    })
                    chunk.page_numbers = [page_num]
                    all_chunks.append(chunk)

        # Re-assign chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"ctx_{i}"

        logger.info(f"Context-aware chunking produced {len(all_chunks)} chunks")
        return all_chunks

    def _extract_sections(self, text: str) -> list[tuple[str | None, str]]:
        """
        Extract sections from text based on header detection.

        Returns list of (section_title, section_text) tuples.
        """
        lines = text.split("\n")
        sections = []
        current_title = None
        current_lines = []

        for line in lines:
            detected_title = self._detect_header(line)

            if detected_title:
                # Save previous section
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                current_title = detected_title
                current_lines = []
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        if not sections:
            sections = [(None, text)]

        return sections

    def _detect_header(self, line: str) -> str | None:
        """Detect if a line is a header/section title."""
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return None

        for pattern in self._header_patterns:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                # Return the captured group or the whole match
                groups = match.groups()
                title = groups[-1] if groups and groups[-1] else stripped
                return title.strip()

        return None

    def _update_hierarchy(
        self, hierarchy: list[str], new_title: str
    ) -> list[str]:
        """Update section hierarchy with a new title."""
        # Simple heuristic: if new title looks like a sub-section, append
        # otherwise, reset to top level
        if len(hierarchy) >= 3:
            hierarchy = [hierarchy[0], new_title]
        else:
            hierarchy.append(new_title)
        return hierarchy[-3:]  # Keep max 3 levels
