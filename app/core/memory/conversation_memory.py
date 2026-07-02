"""
Conversation Memory - Session-based conversation history management.

Supports:
- Per-session message history
- Context window management with sliding window
- Conversation summarization for long sessions
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history per session.

    Features:
    - Stores messages in-memory keyed by session_id
    - Sliding window to limit context size
    - LLM-based summarization for long conversations
    - Extracts context for follow-up queries
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        max_messages: int = 20,
        max_context_messages: int = 6,
    ):
        self.llm = llm_manager
        self.max_messages = max_messages
        self.max_context_messages = max_context_messages
        self._sessions: Dict[str, List[Message]] = defaultdict(list)
        self._summaries: Dict[str, str] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a message to the session history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._sessions[session_id].append(msg)

        # Trim if exceeding max
        if len(self._sessions[session_id]) > self.max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self.max_messages:]

        logger.debug(
            f"Session {session_id}: added {role} message "
            f"(total: {len(self._sessions[session_id])})"
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the full message history for a session."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._sessions.get(session_id, [])
        ]

    def get_context_window(self, session_id: str) -> str:
        """
        Get a formatted context window for the current session.

        Returns the last N messages as formatted text for injection
        into the query pipeline.
        """
        messages = self._sessions.get(session_id, [])
        if not messages:
            return ""

        recent = messages[-self.max_context_messages:]
        context_parts = []

        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")

        return "\n".join(context_parts)

    async def get_contextualized_query(
        self, session_id: str, current_query: str
    ) -> str:
        """
        Resolve references in the current query using conversation history.

        Example:
        - History: "Tell me about chapter 3"
        - Follow-up: "What does it say about pricing?"
        - Resolved: "What does chapter 3 say about pricing?"
        """
        context = self.get_context_window(session_id)
        if not context:
            return current_query

        prompt = f"""Given the conversation history and the user's latest query, 
rewrite the query to be self-contained (resolve all pronouns and references).

Conversation history:
{context}

Latest query: "{current_query}"

Rewrite the query to be fully self-contained. Return ONLY the rewritten query."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a query resolver. Make queries self-contained.",
                temperature=0.1,
                max_tokens=256,
            )
            resolved = response.strip().strip('"\'')
            logger.info(f"Contextualized query: '{current_query}' → '{resolved}'")
            return resolved
        except Exception as e:
            logger.warning(f"Query contextualization failed: {e}")
            return current_query

    async def summarize_session(self, session_id: str) -> str:
        """Generate a summary of the session conversation."""
        messages = self._sessions.get(session_id, [])
        if not messages:
            return "No conversation history."

        conversation = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages
        )

        prompt = f"""Summarize the following conversation in 2-3 sentences:

{conversation[:3000]}

Summary:"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a conversation summarizer.",
                temperature=0.1,
                max_tokens=256,
            )
            summary = response.strip()
            self._summaries[session_id] = summary
            return summary
        except Exception as e:
            logger.warning(f"Session summary failed: {e}")
            return "Unable to generate summary."

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)
        logger.info(f"Cleared session: {session_id}")

    def get_session_ids(self) -> List[str]:
        """Get all active session IDs."""
        return list(self._sessions.keys())

    def get_session_stats(self, session_id: str) -> Dict:
        """Get stats for a session."""
        messages = self._sessions.get(session_id, [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "user_messages": sum(1 for m in messages if m.role == "user"),
            "assistant_messages": sum(1 for m in messages if m.role == "assistant"),
            "has_summary": session_id in self._summaries,
        }
