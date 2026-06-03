"""Shared test fixtures for Agentic Memory OS tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.config import AgentOSConfig
from agent_os.models import Message, Role, SemanticAtom


@pytest.fixture
def config() -> AgentOSConfig:
    """Create a test configuration with dummy credentials."""
    return AgentOSConfig(
        gemini_api_key="test-api-key-not-real",
        kernel_model="gemini-3.5-flash",
        router_model="gemini-3.1-flash-lite",
        elasticsearch_url="http://localhost:9200",
        elasticsearch_api_key="test-elastic-key",
        elastic_index="test_agent_os_memory",
        compression_threshold=3,  # Lower threshold for faster tests
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create a list of sample messages for testing."""
    return [
        Message(role=Role.USER, content="Hello, I'm working on a Python project"),
        Message(role=Role.ASSISTANT, content="Great! What kind of Python project?"),
        Message(role=Role.USER, content="It's a web scraper using BeautifulSoup"),
        Message(role=Role.ASSISTANT, content="Nice choice! BeautifulSoup is great for parsing HTML."),
        Message(role=Role.USER, content="I'm having trouble with async requests"),
        Message(role=Role.ASSISTANT, content="You might want to use aiohttp for async HTTP requests."),
    ]


@pytest.fixture
def sample_atom() -> SemanticAtom:
    """Create a sample semantic atom for testing."""
    return SemanticAtom(
        summary="User is building a Python web scraper using BeautifulSoup and needs help with async requests. Suggested aiohttp.",
        keywords=["python", "web scraper", "beautifulsoup", "async", "aiohttp"],
        source_message_ids=["msg-1", "msg-2", "msg-3"],
        source_message_count=3,
        session_id="test-session-123",
    )


@pytest.fixture
def mock_genai_client():
    """Create a mock google-genai client."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Set up the async generate_content mock
        mock_response = MagicMock()
        mock_response.text = "This is a test response from the LLM."
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        yield mock_client


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    mock = AsyncMock()
    mock.is_connected = True
    mock.search = AsyncMock(return_value=[])
    mock.index_document = AsyncMock(return_value={"result": "created"})
    mock.call_tool = AsyncMock(return_value={})
    return mock
