"""Tests for the SystemKernel module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.kernel import SystemKernel
from agent_os.models import Message, RetrievedContext, Role, SemanticAtom


class TestSystemKernel:
    """Tests for the SystemKernel class."""

    @pytest.fixture
    def kernel(self, config, mock_genai_client):
        """Create a SystemKernel with mocked dependencies."""
        with patch("agent_os.kernel.genai.Client", return_value=mock_genai_client):
            return SystemKernel(config)

    @pytest.mark.asyncio
    async def test_generate_simple_message(self, kernel, mock_genai_client):
        """Kernel should generate a response for a simple message."""
        result = await kernel.generate("Hello!")
        assert result == "This is a test response from the LLM."
        mock_genai_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_recent_messages(self, kernel, mock_genai_client, sample_messages):
        """Kernel should include recent messages in the API call."""
        result = await kernel.generate(
            "Tell me more",
            recent_messages=sample_messages[:2],
        )
        assert result is not None

        # Verify the API was called with contents that include recent messages
        call_args = mock_genai_client.aio.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[0]
        # Recent messages + current message
        assert len(contents) >= 3  # at least 2 recent + 1 current

    @pytest.mark.asyncio
    async def test_generate_with_retrieved_context(self, kernel, mock_genai_client, sample_atom):
        """Kernel should inject retrieved context into the prompt."""
        context = RetrievedContext(atoms=[sample_atom])

        result = await kernel.generate(
            "What was I working on?",
            retrieved_context=context,
        )
        assert result is not None

        # Verify the API was called — context should add preamble messages
        call_args = mock_genai_client.aio.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[0]
        # Context preamble (user + model) + current message = 3
        assert len(contents) >= 3

    @pytest.mark.asyncio
    async def test_generate_with_empty_context(self, kernel, mock_genai_client):
        """Kernel should handle empty retrieved context gracefully."""
        empty_context = RetrievedContext()

        result = await kernel.generate(
            "What's 2+2?",
            retrieved_context=empty_context,
        )
        assert result is not None

        # Should not add context preamble for empty context
        call_args = mock_genai_client.aio.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[0]
        assert len(contents) == 1  # Just the current message

    @pytest.mark.asyncio
    async def test_generate_handles_empty_response(self, kernel, mock_genai_client):
        """Kernel should raise GenerationError on empty LLM response."""
        from agent_os.exceptions import GenerationError

        mock_response = MagicMock()
        mock_response.text = ""
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with pytest.raises(GenerationError, match="empty response"):
            await kernel.generate("Hello!")

    @pytest.mark.asyncio
    async def test_generate_handles_api_error(self, kernel, mock_genai_client):
        """Kernel should wrap API errors in GenerationError."""
        from agent_os.exceptions import GenerationError

        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("API quota exceeded")
        )

        with pytest.raises(GenerationError, match="API quota exceeded"):
            await kernel.generate("Hello!")
