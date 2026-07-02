"""
Query Routes - RAG query, streaming, agentic, and evaluation endpoints.
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import (
    get_memory,
    get_query_pipeline,
)
from app.api.schemas import (
    ConversationHistoryResponse,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    SourceInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with the full pipeline.

    Pipeline stages:
    1. Conversation context resolution
    2. Query rewriting
    3. Hybrid retrieval (vector + BM25)
    4. Cross-encoder re-ranking
    5. Context compression
    6. LLM answer generation
    7. Guardrails validation
    8. (Optional) RAGAS evaluation
    """
    try:
        pipeline = get_query_pipeline()
        result = await pipeline.query(
            question=request.question,
            session_id=request.session_id,
            doc_id_filter=request.doc_id,
            enable_rewrite=request.enable_rewrite,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            search_query=result.get("search_query"),
            validation=result.get("validation", {}),
            evaluation=result.get("evaluation", {}),
            pipeline_metadata=result.get("pipeline_metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        ) from e


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    Stream the RAG response token by token.

    Uses Server-Sent Events (SSE) format for real-time streaming.
    """

    async def event_generator():
        try:
            pipeline = get_query_pipeline()
            async for token in pipeline.query_stream(
                question=request.question,
                session_id=request.session_id,
                doc_id_filter=request.doc_id,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session / Memory Routes ───────────────────────────────────


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active conversation sessions."""
    memory = get_memory()
    session_ids = memory.get_session_ids()

    return [SessionInfo(**memory.get_session_stats(sid)) for sid in session_ids]


@router.get(
    "/sessions/{session_id}/history", response_model=ConversationHistoryResponse
)
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    memory = get_memory()
    history = memory.get_history(session_id)

    return ConversationHistoryResponse(
        session_id=session_id,
        messages=history,
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session's history."""
    memory = get_memory()
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
