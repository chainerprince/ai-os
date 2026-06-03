"""Tests for the RouterMMU module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.models import Message, Role
from agent_os.router import RouterMMU


class TestRouterMMU:
    """Tests for the RouterMMU class."""

    @pytest.fixture
    def router(self, config, mock_genai_client, mock_mcp_client):
        """Create a RouterMMU with mocked dependencies."""
        with patch("agent_os.router.genai.Client", return_value=mock_genai_client):
            return RouterMMU(config, mock_mcp_client)

    @pytest.mark.asyncio
    async def test_route_no_context_needed(self, router, mock_genai_client):
        """Router should return empty context when no history is needed."""
        # Configure the mock to return a "no context needed" decision
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "needs_context": False,
            "search_query": None,
            "reasoning": "Simple greeting, no historical context needed",
        })
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await router.route("Hello!")

        assert result.is_empty
        assert len(result.atoms) == 0

    @pytest.mark.asyncio
    async def test_route_context_needed(self, router, mock_genai_client, mock_mcp_client):
        """Router should search for context when the LLM says it's needed."""
        # Configure the mock to return a "context needed" decision
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "needs_context": True,
            "search_query": "python web scraper beautifulsoup",
            "reasoning": "User is asking about previous project work",
        })
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Mock the MCP search to return results
        mock_mcp_client.search = AsyncMock(return_value=[
            {
                "summary": "User is building a web scraper",
                "keywords": ["python", "scraper"],
                "session_id": "old-session",
                "source_message_count": 5,
            }
        ])

        result = await router.route("What was that project I was working on?")

        # Should have attempted a search
        mock_mcp_client.search.assert_called_once_with(
            query="python web scraper beautifulsoup"
        )

    @pytest.mark.asyncio
    async def test_route_with_recent_messages(self, router, mock_genai_client, sample_messages):
        """Router should include recent messages in its routing decision."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "needs_context": False,
            "search_query": None,
            "reasoning": "Current context is sufficient",
        })
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await router.route(
            "Can you help me with that?",
            recent_messages=sample_messages[:4],
        )

        # Verify the LLM was called (with context from recent messages)
        mock_genai_client.aio.models.generate_content.assert_called_once()
        call_args = mock_genai_client.aio.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents") or call_args.args[0]
        assert "Recent conversation" in prompt

    @pytest.mark.asyncio
    async def test_route_handles_invalid_json(self, router, mock_genai_client):
        """Router should fail open when the LLM returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON at all"
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Should fail open — return empty context, not crash
        result = await router.route("What was that thing?")
        assert result.is_empty

    @pytest.mark.asyncio
    async def test_route_handles_api_failure(self, router, mock_genai_client):
        """Router should fail open when the API call fails."""
        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        # Should fail open
        result = await router.route("What was that thing?")
        assert result.is_empty

    def test_parse_decision_with_code_fence(self, router):
        """Router should handle JSON wrapped in markdown code fences."""
        raw = '```json\n{"needs_context": true, "search_query": "test", "reasoning": "test"}\n```'
        decision = router._parse_decision(raw)
        assert decision.needs_context is True
        assert decision.search_query == "test"
