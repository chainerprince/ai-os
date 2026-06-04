"""
Pydantic data models for Agentic Memory OS.

Defines the core data structures that flow through the system:
messages, semantic atoms, routing decisions, and search results.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message role in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_genai_content(self) -> dict[str, str]:
        """Convert to the format expected by google-genai SDK."""
        return {"role": self.role.value, "parts": [{"text": self.content}]}


class SemanticAtom(BaseModel):
    """A compressed memory unit created by the MemoryDaemon.

    Represents a summary of N messages, enriched with metadata
    for effective hybrid search (semantic + keyword) in Elasticsearch.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str = Field(description="Compressed summary of the conversation chunk")
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords for BM25 keyword search",
    )
    source_message_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the original messages that were compressed",
    )
    source_message_count: int = Field(
        default=0,
        description="Number of original messages compressed into this atom",
    )
    session_id: str = Field(description="ID of the session this atom belongs to")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    time_range_start: datetime | None = Field(
        default=None,
        description="Timestamp of the earliest message in this atom",
    )
    time_range_end: datetime | None = Field(
        default=None,
        description="Timestamp of the latest message in this atom",
    )

    def to_elastic_document(self) -> dict:
        """Convert to a flat dict suitable for Elasticsearch indexing."""
        return {
            "id": self.id,
            "summary": self.summary,
            "keywords": self.keywords,
            "source_message_ids": self.source_message_ids,
            "source_message_count": self.source_message_count,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "time_range_start": (
                self.time_range_start.isoformat() if self.time_range_start else None
            ),
            "time_range_end": (self.time_range_end.isoformat() if self.time_range_end else None),
        }


class RoutingDecision(BaseModel):
    """The RouterMMU's decision on whether to fetch historical context."""

    needs_context: bool = Field(
        description="Whether the current prompt requires historical context",
    )
    search_query: str | None = Field(
        default=None,
        description="The search query to use for context retrieval (if needed)",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the routing decision",
    )


class RetrievedContext(BaseModel):
    """Context retrieved from Elasticsearch via the MCP server."""

    atoms: list[SemanticAtom] = Field(default_factory=list)
    raw_results: list[dict] = Field(
        default_factory=list,
        description="Raw search results from Elasticsearch",
    )

    @property
    def combined_summary(self) -> str:
        """Combine all atom summaries into a single context string."""
        if not self.atoms:
            return ""
        summaries = [
            f"[Memory from {atom.created_at.strftime('%Y-%m-%d %H:%M')}]: {atom.summary}"
            for atom in self.atoms
        ]
        return "\n\n".join(summaries)

    @property
    def is_empty(self) -> bool:
        """Check if any context was retrieved."""
        return len(self.atoms) == 0
