from app.api.schemas import HealthResponse
from fastapi import APIRouter


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider="test",
        embedding_model="test",
        total_vectors=0,
        total_documents=0,
    )

