"""
Summary Chunker - Creates summary chunks for non-text elements.

Creates summary chunks for images and tables found in PDF content.
"""

import logging

from app.api.schemas import PdfData
from app.config import settings
from app.core.chunking.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


class SummaryChunker:
    """
    Splits documents while preserving structural context.

    Features:
    - Creates summary chunks for images and tables
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_non_text_elements(
        self, pages: list[PdfData], base_metadata: dict
    ) -> list[Chunk]:
        """Create chunks for images and tables."""
        chunks = []
        for page in pages:
            if page.type == "image":
                for img in page.data.get("images", []):
                    chunks.append(
                        Chunk(
                            content=f"[Summary of Image: {img['text']}]",
                            metadata={
                                **base_metadata,
                                "page_number": page.page_number,
                                "element_type": "image",
                                "img_id": img["img_id"],
                            },
                            chunk_id=f"img_{img['img_id']}",
                            page_numbers=[page.page_number],
                        )
                    )
            elif page.type == "table":
                for tbl in page.data.get("tables", []):
                    chunks.append(
                        Chunk(
                            content=f"[Summary of Table: {tbl['text']}]",
                            metadata={
                                **base_metadata,
                                "page_number": page.page_number,
                                "element_type": "table",
                                "tbl_id": tbl["tbl_id"],
                            },
                            chunk_id=f"tbl_{tbl['tbl_id']}",
                            page_numbers=[page.page_number],
                        )
                    )
        return chunks
