"""Tests for the MemoryDaemon module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon import MemoryDaemon
from agent_os.models import Message, Role


class TestMemoryDaemon:
    """Tests for the MemoryDaemon class."""

    @pytest.fixture
    def mock_es_client(self):
        """Create a mock AsyncElasticsearch client."""
        mock_es = AsyncMock()
        mock_es.index = AsyncMock(return_value={"result": "created", "_id": "test-id"})
        mock_es.close = AsyncMock()
        return mock_es

    @pytest.fixture
    def daemon(self, config, mock_genai_client, mock_mcp_client, mock_es_client):
        """Create a MemoryDaemon with mocked dependencies."""
        with (
            patch("agent_os.daemon.genai.Client", return_value=mock_genai_client),
            patch("agent_os.daemon.AsyncElasticsearch", return_value=mock_es_client),
        ):
            d = MemoryDaemon(config, mock_mcp_client, session_id="test-session")
            d._es = mock_es_client  # Ensure the mock is used
            return d

    def test_add_message_increments_buffer(self, daemon, sample_messages):
        """Adding messages should increase the buffer size."""
        assert daemon.buffer_size == 0

        daemon.add_message(sample_messages[0])
        assert daemon.buffer_size == 1

        daemon.add_message(sample_messages[1])
        assert daemon.buffer_size == 2

    def test_initial_state(self, daemon):
        """Daemon should start in a non-running state with empty buffer."""
        assert not daemon.is_running
        assert daemon.buffer_size == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, daemon):
        """Flushing an empty buffer should return None."""
        result = await daemon.flush()
        assert result is None

    @pytest.mark.asyncio
    async def test_flush_with_messages(self, daemon, mock_genai_client, mock_es_client, sample_messages):
        """Flushing with messages should compress and store a semantic atom."""
        # Configure the mock LLM to return a compression result
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "User discussed Python web scraping with BeautifulSoup",
            "keywords": ["python", "web scraping", "beautifulsoup"],
        })
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Add messages to the buffer
        for msg in sample_messages[:3]:
            daemon.add_message(msg)

        assert daemon.buffer_size == 3

        # Flush
        atom = await daemon.flush()

        assert atom is not None
        assert atom.summary == "User discussed Python web scraping with BeautifulSoup"
        assert "python" in atom.keywords
        assert atom.source_message_count == 3
        assert atom.session_id == "test-session"

        # Buffer should be cleared
        assert daemon.buffer_size == 0

        # Should have stored the atom via direct ES client
        mock_es_client.index.assert_called_once()

    @pytest.mark.asyncio
    async def test_compression_error_handling(self, daemon, mock_genai_client, sample_messages):
        """Daemon should raise CompressionError on LLM failure."""
        from agent_os.exceptions import CompressionError

        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("API unavailable")
        )

        for msg in sample_messages[:3]:
            daemon.add_message(msg)

        with pytest.raises(CompressionError, match="API unavailable"):
            await daemon.flush()

    @pytest.mark.asyncio
    async def test_storage_error_handling(self, daemon, mock_genai_client, mock_es_client, sample_messages):
        """Daemon should raise StorageError when Elastic write fails."""
        from agent_os.exceptions import StorageError

        # LLM succeeds
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "Test summary",
            "keywords": ["test"],
        })
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # But storage fails via direct ES client
        mock_es_client.index = AsyncMock(
            side_effect=RuntimeError("Elastic connection refused")
        )

        for msg in sample_messages[:3]:
            daemon.add_message(msg)

        with pytest.raises(StorageError, match="Elastic connection refused"):
            await daemon.flush()

    @pytest.mark.asyncio
    async def test_start_creates_task(self, daemon):
        """Starting the daemon should create a background asyncio task."""
        daemon.start()
        assert daemon.is_running

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, daemon):
        """Stopping the daemon should set running to False."""
        daemon.start()
        assert daemon.is_running

        await daemon.stop()
        assert not daemon.is_running
