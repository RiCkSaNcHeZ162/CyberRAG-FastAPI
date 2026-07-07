import logging
import re
import uuid
from pathlib import Path
from typing import Any

import pymupdf4llm

from app.api.schemas import PdfData
from app.config import settings
from app.core.chunking.context_aware import ContextAwareChunker
from app.core.chunking.recursive_chunker import RecursiveChunker
from app.core.chunking.semantic_chunker import Chunk, SemanticChunker
from app.core.chunking.summary_chunker import SummaryChunker
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.media.media_handler import MediaHandler
from app.core.summary.summary import PDFSummary
from app.core.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    PDF → Extract Text → Chunk → Embed → Store
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        embedding_manager: EmbeddingManager,
        pdf_summary: PDFSummary,
        media_handler: MediaHandler,
        chunking_strategy: str = settings.CHUNKING_STRATEGY,
    ):
        self.faiss_store = faiss_store
        self.embedding_manager = embedding_manager
        self.chunking_strategy = chunking_strategy
        self.pdf_summary = pdf_summary
        self.media_handler = media_handler
        self.previous_page_number = None

        # Initialize chunkers
        self._semantic_chunker = None
        self._recursive_chunker = RecursiveChunker()
        self._context_aware_chunker = ContextAwareChunker()
        self._summary_chunker = SummaryChunker()

    def _get_semantic_chunker(self) -> SemanticChunker:
        """Lazy-load semantic chunker (shares embedding model)."""
        if self._semantic_chunker is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._semantic_chunker = SemanticChunker(embedding_model=model)
        return self._semantic_chunker

    async def ingest_pdf(
        self,
        file_path: str,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a PDF file into the RAG system.

        Args:
            file_path: Path to the PDF file.
            doc_id: Optional document ID. Auto-generated if not provided.
            chunking_strategy: Override chunking strategy for this document.

        Returns:
            Ingestion result dict with stats.
        """
        doc_id = doc_id or str(uuid.uuid4())[:8]
        strategy = chunking_strategy or self.chunking_strategy
        file_path = Path(file_path)

        logger.info(
            f"Starting ingestion: {file_path.name} (doc_id={doc_id}, strategy={strategy})"
        )

        # Step 1: Extract text from PDF
        pages: list[PdfData] = await self._extract_and_summarize_pdf(str(file_path))
        if not pages:
            raise ValueError(f"Could not extract text from {file_path.name}")

        total_text_length = sum(len(p.data["text"]) for p in pages if p.type == "text")
        logger.info(f"Extracted {len(pages)} pages ({total_text_length} chars)")

        # Step 2: Chunk the text
        chunks = self._chunk_text(
            pages,
            strategy,
            {
                "doc_id": doc_id,
                "source_file": file_path.name,
                "total_pages": len(pages),
            },
        )
        if not chunks:
            raise ValueError("Chunking produced no chunks")

        logger.info(f"Created {len(chunks)} chunks using '{strategy}' strategy")

        # Step 3: Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = self.embedding_manager.embed_documents(
            chunk_texts, show_progress=True
        )

        # Step 4: Store in FAISS
        metadata_list = [
            {**c.metadata, "chunk_id": c.chunk_id, "page_numbers": c.page_numbers}
            for c in chunks
        ]

        n_added = self.faiss_store.add(
            vectors=embeddings,
            texts=chunk_texts,
            metadata_list=metadata_list,
            doc_id=doc_id,
        )

        # Step 5: Save to disk
        self.faiss_store.save()

        result = {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "total_pages": len(pages),
            "total_characters": total_text_length,
            "chunks_created": len(chunks),
            "vectors_stored": n_added,
            "chunking_strategy": strategy,
            "embedding_dimension": self.embedding_manager.embedding_dimension,
            "status": "success",
        }

        logger.info(f"Ingestion complete: {result}")
        return result

    async def _extract_and_summarize_pdf(self, file_path: str) -> list[PdfData]:
        try:
            # sort the images tables and text from markdown
            processed_pages: list[PdfData] = self._process_pages(file_path=file_path)
            await self.pdf_summary.summarize_pdf_pages(processed_pages)
            return processed_pages
        except Exception as e:
            logger.error(f"Error during PDF text extraction: {e}")
            return []

    def _process_pages(self, file_path: str) -> list[PdfData]:
        print(file_path)
        processed_pages: list[PdfData] = []
        pages = pymupdf4llm.to_markdown(
            file_path,
            page_chunks=True,  # per-page dicts
            embed_images=True,  # base64 images inline
            table_strategy="lines_strict",
            header=False,  # skip repetitive headers
            footer=False,  # skip repetitive footers
        )

        for page in pages:
            page_num = page["metadata"]["page_number"]
            filename = page["metadata"]["file_path"]
            full_text = page["text"]
            # ── 1. PLAIN TEXT chunks ──────────────────────────────
            # Remove table and image sections from text for clean text chunk

            text_only = re.sub(r"!\[.*?\]\(data:image.*?\)", "", full_text)
            text_only = re.sub(r"\|.*\|", "", text_only).strip()
            if text_only:
                processed_pages.append(
                    PdfData(
                        data={"text": text_only},
                        type="text",
                        page_number=page_num,
                        filename=filename,
                    )
                )

            # ── 2. TABLES ─────────────────────────────────────────
            tables: PdfData = self._extract_markdown_tables(
                full_text,
                page_num,
                filename,
            )  ## contains all tables on the page, each with a unique doc_id
            if (
                hasattr(tables, "data")
                and "tables" in tables.data
                and len(tables.data["tables"])
            ):
                processed_pages.append(tables)

            # ── 3. IMAGES ─────────────────────────────────────────
            images: PdfData = self._extract_base64_images(
                full_text,
                page_num,
                filename,
            )  ## contains all images on the page, each with a unique doc_id
            if (
                hasattr(images, "data")
                and "images" in images.data
                and len(images.data["images"])
            ):
                processed_pages.append(images)

        return processed_pages

    def _extract_base64_images(
        self,
        markdown_text: str,
        page_num: int,
        filename: str,
    ) -> PdfData:
        """Pull out base64 images embedded in markdown."""
        pattern = r"!\[.*?\]\(data:image/(.*?);base64,(.*?)\)"
        matches = re.findall(pattern, markdown_text)
        data = [
            {"format": m[0], "data": m[1], "img_id": str(uuid.uuid4())} for m in matches
        ]
        if len(data):
            # save images for future querying
            self.media_handler.save_images_to_disk(
                data,
            )
            return PdfData(
                data={"images": data},
                type="image",
                page_number=page_num,
                filename=filename,
            )
        else:
            return {}

    def _extract_markdown_tables(
        self,
        markdown_text: str,
        page_num: int,
        filename: str,
    ) -> PdfData:
        """Pull out markdown table blocks."""
        tables = []
        lines = markdown_text.split("\n")
        current_table = []

        for line in lines:
            if "|" in line:
                current_table.append(line)
            else:
                if len(current_table) > 2:  # at least header + separator + 1 row
                    tbl_id = str(uuid.uuid4())
                    tables.append({"data": "\n".join(current_table), "tbl_id": tbl_id})
                current_table = []
        if len(tables):
            self.media_handler.save_tables_to_disk(tables)
            return PdfData(
                data={"tables": tables},
                type="table",
                page_number=page_num,
                filename=filename,
            )
        else:
            return {}

    def _chunk_text(
        self,
        pages: list[PdfData],
        strategy: str,
        base_metadata: dict,
    ) -> list[Chunk]:
        """Chunk extracted text using the specified strategy."""
        pages_to_chunk = [
            p for p in pages if p.type == "text" and p.data["text"].strip()
        ]
        if strategy == "context_aware":
            return self._context_aware_chunker.chunk_pages(
                pages_to_chunk, base_metadata
            )

        # For semantic and recursive, concatenate all text first
        full_text = "\n\n".join(
            f"[Page {p.page_number}]\n{p.data['text']}" for p in pages_to_chunk
        )

        if strategy == "semantic":
            chunker = self._get_semantic_chunker()
            chunks: list[Chunk] = chunker.chunk(full_text, base_metadata)
        elif strategy == "recursive":
            chunks: list[Chunk] = self._recursive_chunker.chunk(
                full_text, base_metadata
            )
        else:
            # Default to recursive
            chunks: list[Chunk] = self._recursive_chunker.chunk(
                full_text, base_metadata
            )

        # Attempt to assign page numbers to chunks
        for chunk in chunks:
            chunk.page_numbers = self._infer_page_numbers(chunk.content, pages_to_chunk)
        self.previous_page_number = (
            None  # reset previous page number after processing all chunks
        )

        # handle photos and tables as separate chunks
        separate_chunks = self._summary_chunker._chunk_non_text_elements(
            pages, base_metadata
        )
        chunks.extend(separate_chunks)
        return chunks

    def _infer_page_numbers(self, chunk_text: str, pages: list[PdfData]) -> list[int]:
        """Try to figure out which pages a chunk came from."""
        page_numbers = []

        # Check for page markers
        import re

        markers = re.findall(r"\[Page (\d+)\]", chunk_text)
        match = re.search(r"\[Page (\d+)\]\s*$", chunk_text)
        if markers:
            page_numbers = list(set(int(m) for m in markers))
            if match and len(page_numbers) > 1:
                self.previous_page_number = page_numbers.pop()
            elif match:
                page_numbers = [self.previous_page_number]
            else:
                self.previous_page_number = page_numbers[-1]
        else:
            # If no markers, use previous page number if available
            if self.previous_page_number is not None:
                page_numbers = [self.previous_page_number]
