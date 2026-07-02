"""
Query Rewriter - LLM-powered query expansion and reformulation.

Techniques:
1. Query Expansion: Generates multiple alternative phrasings
2. Multi-Query Retrieval: Retrieves from each query variant, deduplicates
3. HyDE: Hypothetical Document Embedding
"""

import logging

from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Improves retrieval by reformulating user queries using an LLM.

    Handles vague, ambiguous, or poorly-formed queries by:
    - Expanding with synonyms and related terms
    - Generating multiple query perspectives
    - Creating hypothetical answer embeddings (HyDE)
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    async def expand_query(self, query: str, n_expansions: int = 3) -> list[str]:
        """
        Generate multiple alternative phrasings of the query.

        Args:
            query: Original user query.
            n_expansions: Number of alternative queries to generate.

        Returns:
            List of expanded queries (including original).
        """
        prompt = f"""Given the following search query, generate {n_expansions} alternative phrasings 
that capture the same intent but use different words, perspectives, or levels of specificity.

Original query: "{query}"

Return ONLY the alternative queries, one per line, numbered 1-{n_expansions}.
Do not include any explanation or the original query."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a query expansion expert. Generate diverse search queries.",
                temperature=0.7,
                max_tokens=512,
            )

            # Parse numbered responses
            expanded = [query]  # Always include original
            for line in response.strip().split("\n"):
                cleaned = line.strip()
                # Remove numbering (1. or 1) prefix)
                if cleaned and cleaned[0].isdigit():
                    cleaned = cleaned.lstrip("0123456789.)- ").strip()
                if cleaned and len(cleaned) > 5:
                    expanded.append(cleaned)

            logger.info(f"Query expansion: '{query}' → {len(expanded)} variants")
            return expanded[: n_expansions + 1]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return [query]

    async def rewrite_query(self, query: str, conversation_context: str = "") -> str:
        """
        Rewrite a vague or conversational query into a clear search query.

        Args:
            query: Original user query.
            conversation_context: Previous conversation for context.

        Returns:
            Rewritten, clearer query.
        """
        context_section = ""
        if conversation_context:
            context_section = f"\n\nConversation context:\n{conversation_context}"

        prompt = f"""Rewrite the following user query into a clear, specific search query 
that will retrieve the most relevant information from a document.

If the query references previous conversation context, resolve all pronouns and references.
{context_section}

User query: "{query}"

Return ONLY the rewritten query, nothing else."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a search query optimization expert.",
                temperature=0.1,
                max_tokens=256,
            )
            rewritten = response.strip().strip("\"'")
            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Using original query.")
            return query

    async def generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE - Generate a hypothetical answer to embed for retrieval.

        Instead of embedding the question, embed a hypothetical answer.
        This often captures more relevant semantic content for retrieval.

        Args:
            query: User query.

        Returns:
            Hypothetical answer text.
        """
        prompt = f"""Write a brief, factual paragraph that would be the ideal answer to 
the following question. Write as if you are quoting from a reference document.

Question: "{query}"

Write ONLY the hypothetical answer paragraph, nothing else."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a document author writing reference material.",
                temperature=0.3,
                max_tokens=512,
            )
            hypothesis = response.strip()
            logger.info(f"HyDE generated {len(hypothesis)} chars for query: '{query}'")
            return hypothesis

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original query.")
            return query
