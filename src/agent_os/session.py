"""
Session — The Orchestrator of Agentic Memory OS.

The user-facing entry point that ties together the SystemKernel, RouterMMU,
and MemoryDaemon into a coherent conversation session. Users interact with
this class to chat with the system.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from agent_os.daemon import MemoryDaemon
from agent_os.exceptions import SessionNotStartedError
from agent_os.kernel import SystemKernel
from agent_os.logging import get_logger, setup_logging
from agent_os.mcp_client import ElasticMCPClient
from agent_os.models import Message, Role, SemanticAtom
from agent_os.router import RouterMMU

if TYPE_CHECKING:
    from agent_os.config import AgentOSConfig

logger = get_logger("session")

# How many recent messages to send to the kernel for short-term context
RECENT_MESSAGE_WINDOW = 10


class Session:
    """The Orchestrator — ties together Kernel, Router, and Daemon.

    This is the main class that SDK users interact with. It manages the
    full lifecycle of a conversation:

    1. Start: initializes the MCP connection, kernel, router, and daemon
    2. Chat: routes user messages through the MMU → Kernel pipeline
    3. Stop: flushes the daemon and closes connections

    Usage:
        config = AgentOSConfig()
        session = Session(config)

        async with session:
            response = await session.chat("Hello!")
            response = await session.chat("What did I say earlier?")
    """

    def __init__(self, config: AgentOSConfig, session_id: str | None = None) -> None:
        self._config = config
        self._session_id = session_id or str(uuid.uuid4())
        self._messages: list[Message] = []
        self._started = False

        # Components — initialized in start()
        self._mcp_client = ElasticMCPClient(config)
        self._kernel: SystemKernel | None = None
        self._router: RouterMMU | None = None
        self._daemon: MemoryDaemon | None = None

        # Context manager for MCP connection
        self._mcp_context = None

        # Metrics for System Monitor
        self.total_tokens_saved = 0
        self.total_atoms_swapped = 0
        self.last_turn_metrics = {
            "latency_sec": 0.0,
            "page_fault": False,
            "daemon_buffer": 0,
        }

        logger.info("session_created", session_id=self._session_id)

    @property
    def session_id(self) -> str:
        """The unique ID for this session."""
        return self._session_id

    @property
    def messages(self) -> list[Message]:
        """All messages in the current session."""
        return list(self._messages)

    @property
    def message_count(self) -> int:
        """Number of messages in the session."""
        return len(self._messages)

    @property
    def is_started(self) -> bool:
        """Whether the session has been started."""
        return self._started

    async def start(self) -> None:
        """Initialize all components and start the daemon.

        Sets up the MCP connection, creates the Kernel, Router, and Daemon,
        and starts the background compression task.
        """
        if self._started:
            logger.warning("session_already_started", session_id=self._session_id)
            return

        # Initialize logging
        setup_logging()

        logger.info("session_starting", session_id=self._session_id)

        # Connect to the Elastic MCP server
        self._mcp_context = self._mcp_client.connect()
        await self._mcp_context.__aenter__()

        # Initialize components
        self._kernel = SystemKernel(self._config)
        self._router = RouterMMU(self._config, self._mcp_client)
        self._daemon = MemoryDaemon(
            self._config,
            self._mcp_client,
            self._session_id,
            on_compress_success=self._on_atom_stored,
        )

        # Start the background daemon
        self._daemon.start()

        self._started = True
        logger.info("session_started", session_id=self._session_id)

    async def stop(self) -> None:
        """Gracefully shut down the session.

        Stops the daemon (flushing remaining messages) and closes
        the MCP connection.
        """
        if not self._started:
            return

        logger.info("session_stopping", session_id=self._session_id)

        # Stop the daemon (will flush remaining messages)
        if self._daemon:
            await self._daemon.stop()

        # Close the MCP connection
        if self._mcp_context:
            await self._mcp_context.__aexit__(None, None, None)
            self._mcp_context = None

        self._started = False
        logger.info(
            "session_stopped",
            session_id=self._session_id,
            total_messages=len(self._messages),
        )

    async def chat(self, user_input: str) -> str:
        """Send a message and get a response.

        This is the main interaction method. It:
        1. Records the user message
        2. Routes through the MMU to check if context is needed
        3. Generates a response via the Kernel
        4. Records the assistant response
        5. Feeds both messages to the Daemon's buffer

        Args:
            user_input: The user's message text.

        Returns:
            The assistant's response text.

        Raises:
            SessionNotStartedError: If start() hasn't been called.
        """
        if not self._started:
            raise SessionNotStartedError()

        assert self._kernel is not None
        assert self._router is not None
        assert self._daemon is not None

        import time

        start_time = time.time()

        # Record user message
        user_message = Message(role=Role.USER, content=user_input)
        self._messages.append(user_message)

        logger.info(
            "chat_turn_started",
            session_id=self._session_id,
            message_number=len(self._messages),
        )

        # Step 1: Route — does this message need historical context?
        recent = self._get_recent_messages()
        retrieved_context = await self._router.route(user_input, recent)
        page_fault = not retrieved_context.is_empty

        # Step 2: Generate — produce a response with the Kernel
        response_text = await self._kernel.generate(
            user_message=user_input,
            recent_messages=recent,
            retrieved_context=retrieved_context,
        )

        # Record assistant message
        assistant_message = Message(role=Role.ASSISTANT, content=response_text)
        self._messages.append(assistant_message)

        # Step 3: Buffer — feed messages to the daemon for future compression
        self._daemon.add_message(user_message)
        self._daemon.add_message(assistant_message)

        # Update metrics
        self.last_turn_metrics = {
            "latency_sec": time.time() - start_time,
            "page_fault": page_fault,
            "daemon_buffer": self._daemon.buffer_size,
        }

        logger.info(
            "chat_turn_completed",
            session_id=self._session_id,
            context_used=page_fault,
            response_length=len(response_text),
            daemon_buffer_size=self._daemon.buffer_size,
        )

        return response_text

    def _on_atom_stored(self, atom: SemanticAtom) -> None:
        """Callback invoked by the MemoryDaemon when messages are swapped to Elastic.

        This implements True Paging: once compressed, messages are evicted from RAM.
        """
        compressed_ids = set(atom.source_message_ids)

        # Estimate token savings (rough heuristic: 1 token ~= 4 chars)
        for msg in self._messages:
            if msg.id in compressed_ids:
                self.total_tokens_saved += len(msg.content) // 4

        # Evict from active RAM
        self._messages = [m for m in self._messages if m.id not in compressed_ids]
        self.total_atoms_swapped += 1

        logger.info("messages_evicted", count=len(compressed_ids), new_ram_size=len(self._messages))

    def _get_recent_messages(self) -> list[Message]:
        """Get the most recent messages for short-term context.

        Returns the last N messages (configured by RECENT_MESSAGE_WINDOW),
        excluding the message currently being processed.
        """
        return self._messages[-(RECENT_MESSAGE_WINDOW + 1) : -1] if len(self._messages) > 1 else []

    async def __aenter__(self) -> Session:
        """Async context manager — start the session."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager — stop the session."""
        await self.stop()
