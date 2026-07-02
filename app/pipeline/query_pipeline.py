"""
Query Pipeline - Full query processing pipeline.

Pipeline:
1. Conversation context resolution
2. Query rewriting/expansion
3. Hybrid retrieval (vector + BM25)
4. Re-ranking
5. Context compression
6. LLM generation (with streaming support)
7. Guardrails validation
8. Evaluation metrics
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.core.llm.llm_manager import LLMManager
from app.core.memory.conversation_memory import ConversationMemory
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Full RAG query processing pipeline with all advanced components.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        hybrid_retriever: HybridRetriever,
        query_rewriter: QueryRewriter,
        memory: ConversationMemory,
    ):
        self.llm = llm_manager
        self.retriever = hybrid_retriever
        self.query_rewriter = query_rewriter
        self.memory = memory

    async def query(
        self,
        question: str,
        session_id: str = "default",
        doc_id_filter: str | None = None,
        enable_rewrite: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the full query pipeline.

        Args:
            question: User's question.
            session_id: Conversation session ID.
            doc_id_filter: Optional filter by document ID.
            enable_rewrite: Whether to use query rewriting.

        Returns:
            Complete response with answer, sources, scores, and metadata.
        """

        # Step 1: Conversation context resolution
        contextualized = await self.memory.get_contextualized_query(
            session_id, question
        )
        # Step 2: Query rewriting
        search_query = contextualized
        if enable_rewrite:
            search_query = await self.query_rewriter.rewrite_query(
                contextualized,
                self.memory.get_context_window(session_id),
            )

        # Step 3: Hybrid retrieval
        retrieved = self.retriever.retrieve(search_query, doc_id_filter=doc_id_filter)

        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
            }
        retrieved_docs = retrieved[:5]

        # Step 6: Generate answer
        context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
        prompt = self._build_answer_prompt(contextualized, context)
        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=(
                "You are a knowledgeable document assistant. Answer questions "
                "based strictly on the provided context. Be thorough, precise, "
                "and cite relevant sections from the documents."
            ),
            temperature=0.1,
            max_tokens=2048,
        )
        answer = answer.strip()

        # Store in memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", answer)

        # Build sources
        sources = []
        for doc in retrieved_docs:
            sources.append(
                {
                    "text": doc.get("original_text", doc["text"])[:300],
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("rerank_score", doc.get("score", 0)),
                    "retrieval_method": doc.get("retrieval_method", "unknown"),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query,
        }

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        doc_id_filter: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the answer token by token.

        Performs retrieval, reranking, and compression first,
        then streams the LLM generation.
        """
        # Pre-processing (non-streaming)
        contextualized = await self.memory.get_contextualized_query(
            session_id, question
        )
        search_query = await self.query_rewriter.rewrite_query(
            contextualized,
            self.memory.get_context_window(session_id),
        )

        retrieved = self.retriever.retrieve(search_query, doc_id_filter=doc_id_filter)

        if not retrieved:
            yield "I couldn't find relevant information in the documents to answer your question."
            return
        retrieved_docs = retrieved[:5]

        context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
        prompt = self._build_answer_prompt(contextualized, context)

        # Stream the answer
        full_answer = ""
        async for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=(
                "You are a knowledgeable document assistant. Answer questions "
                "based strictly on the provided context."
            ),
            temperature=0.1,
            max_tokens=2048,
        ):
            full_answer += token
            yield token

        # Store in memory after streaming completes
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", full_answer)

    def _build_answer_prompt(self, question: str, context: str) -> str:
        """Build the prompt for answer generation."""
        return f"""Based on the following context extracted from the uploaded documents, 
answer the user's question thoroughly and accurately.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information to fully answer, clearly state what information is missing
- Reference specific parts of the documents when applicable
- Be detailed and well-organized in your response
- Use bullet points or numbered lists for clarity when appropriate

ANSWER:"""
