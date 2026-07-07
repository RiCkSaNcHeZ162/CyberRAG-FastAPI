"""
Dependency Injection - Shared component instances for the API.

Provides singleton instances of all RAG components to route handlers.
"""

# ── Document Registry ─────────────────────────────────────────
# In-memory registry of uploaded documents
from functools import lru_cache

from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.llm.llm_manager import LLMManager
from app.core.media.media_handler import MediaHandler
from app.core.memory.conversation_memory import ConversationMemory
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.query_rewriter import QueryRewriter
from app.core.summary.summary import PDFSummary
from app.core.vectorstore.faiss_store import FAISSStore
from app.pipeline.ingestion import IngestionPipeline
from app.pipeline.query_pipeline import QueryPipeline

_document_registry: dict = {}


def get_document_registry() -> dict:
    return _document_registry


@lru_cache
def get_llm_manager() -> LLMManager:
    return LLMManager()


@lru_cache
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@lru_cache
def get_media_handler() -> MediaHandler:
    return MediaHandler()


@lru_cache
def get_pdf_summary() -> PDFSummary:
    return PDFSummary(llm_manager=get_llm_manager())


@lru_cache
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter(llm_manager=get_llm_manager())


@lru_cache
def get_faiss_store() -> FAISSStore:
    em = get_embedding_manager()
    return FAISSStore(collection_name="main", dimension=em.embedding_dimension)


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        faiss_store=get_faiss_store(),
        embedding_manager=get_embedding_manager(),
    )


@lru_cache
def get_memory() -> ConversationMemory:
    return ConversationMemory(llm_manager=get_llm_manager())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        faiss_store=get_faiss_store(),
        embedding_manager=get_embedding_manager(),
        pdf_summary=get_pdf_summary(),
        media_handler=get_media_handler(),
    )


@lru_cache
def get_query_pipeline() -> QueryPipeline:
    return QueryPipeline(
        llm_manager=get_llm_manager(),
        hybrid_retriever=get_hybrid_retriever(),
        query_rewriter=get_query_rewriter(),
        memory=get_memory(),
        media_handler=get_media_handler()
    )
