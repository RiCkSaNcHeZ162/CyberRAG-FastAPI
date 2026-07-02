from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Document Schemas ──────────────────────────────────────────


class DocumentUploadResponse(BaseModel):
    """Response after uploading and processing a PDF."""

    doc_id: str
    file_name: str
    total_pages: int
    total_characters: int
    chunks_created: int
    vectors_stored: int
    chunking_strategy: str
    embedding_dimension: int
    status: str


# ── Health Check ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health status."""

    status: str = "healthy"
    version: str = "1.0.0"
    llm_provider: str
    embedding_model: str
    total_vectors: int = 0
    total_documents: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Markdown Page ─────────────────────────────────────


class MarkdownPage(BaseModel):
    metadata: dict
    toc_items: list[list[int, str, int]]
    page_boxes: list[dict]
    text: str


class PdfData(BaseModel):
    data: dict
    type: str
    page_number: int
    filename: str


class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    upload_time: str
    chunks_count: int
    chunking_strategy: str


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    vectors_deleted: int
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_count: int


# ── Query Schemas ─────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request to query the RAG system."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="The question to ask"
    )
    session_id: str = Field(
        default="default", description="Session ID for conversation memory"
    )
    doc_id: str | None = Field(
        default=None, description="Filter by specific document ID"
    )
    enable_rewrite: bool = Field(default=True, description="Enable query rewriting")


class SourceInfo(BaseModel):
    """Information about a source chunk."""

    text: str
    metadata: dict[str, Any] = {}
    score: float = 0.0
    retrieval_method: str = "unknown"


class QueryResponse(BaseModel):
    """Response from the RAG query pipeline."""

    answer: str
    sources: list[SourceInfo] = []
    search_query: str | None = None
    validation: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    pipeline_metadata: dict[str, Any] = {}


# ── Session Schemas ────────────────────────────────────────────


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    message_count: int
    user_messages: int
    assistant_messages: int
    has_summary: bool


class ConversationHistoryResponse(BaseModel):
    """Conversation history."""

    session_id: str
    messages: list[dict[str, str]]
