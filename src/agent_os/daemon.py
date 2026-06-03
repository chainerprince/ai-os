"""
MemoryDaemon — The Background Worker of Agentic Memory OS.

Monitors the active session's message buffer. When the configured threshold
is reached (default: 10 messages), it compresses the oldest messages into
a "semantic atom" (summary + metadata) and writes it to Elasticsearch.

Uses the Elasticsearch Python client directly for writes (the MCP server
is read-only), and runs as an asyncio background task.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from elasticsearch import AsyncElasticsearch
from google import genai

from agent_os.exceptions import CompressionError, StorageError
from agent_os.logging import get_logger
from agent_os.models import Message, Role, SemanticAtom

if TYPE_CHECKING:
    from agent_os.config import AgentOSConfig
    from agent_os.mcp_client import ElasticMCPClient

logger = get_logger("daemon")

# System prompt for the compression LLM
COMPRESSION_SYSTEM_PROMPT = """\
You are a memory compression agent. Your job is to compress a chunk of conversation
into a concise, information-dense summary that can be used for future context retrieval.

Requirements:
1. Capture ALL important facts, decisions, preferences, and action items.
2. Preserve specific details: names, numbers, dates, code snippets, technical terms.
3. Use clear, searchable language — this summary will be indexed for hybrid search.
4. Respond with ONLY a JSON object:

{
  "summary": "concise but thorough summary of the conversation chunk",
  "keywords": ["keyword1", "keyword2", "keyword3", ...]
}

The keywords should include important nouns, technical terms, and topic identifiers
that would help a search engine find this memory later."""


class MemoryDaemon:
    """The Background Worker — async memory compressor.

    Watches a message buffer and automatically compresses old messages
    into semantic atoms when the threshold is reached. Each atom is then
    indexed into Elasticsearch using the async Python client directly.

    The daemon runs as an asyncio task and can be started/stopped with
    the session lifecycle.
    """

    def __init__(
        self,
        config: AgentOSConfig,
        mcp_client: ElasticMCPClient,
        session_id: str,
    ) -> None:
        self._config = config
        self._mcp_client = mcp_client
        self._session_id = session_id
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._threshold = config.compression_threshold

        # Direct Elasticsearch client for writes
        self._es = AsyncElasticsearch(
            config.elasticsearch_url,
            api_key=config.elasticsearch_api_key,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )

        # Internal state
        self._message_buffer: list[Message] = []
        self._task: asyncio.Task[None] | None = None
        self._buffer_event = asyncio.Event()
        self._running = False

        logger.info(
            "daemon_initialized",
            session_id=session_id,
            threshold=self._threshold,
        )

    @property
    def is_running(self) -> bool:
        """Check if the daemon background task is active."""
        return self._running and self._task is not None and not self._task.done()

    @property
    def buffer_size(self) -> int:
        """Current number of messages in the buffer."""
        return len(self._message_buffer)

    def add_message(self, message: Message) -> None:
        """Add a message to the daemon's buffer.

        When the buffer reaches the threshold, the daemon's background task
        will wake up and compress the messages.

        Args:
            message: The message to buffer.
        """
        self._message_buffer.append(message)
        logger.debug(
            "message_buffered",
            buffer_size=len(self._message_buffer),
            threshold=self._threshold,
        )

        # Signal the background task if we've hit the threshold
        if len(self._message_buffer) >= self._threshold:
            self._buffer_event.set()

    def start(self) -> None:
        """Start the daemon as a background asyncio task."""
        if self.is_running:
            logger.warning("daemon_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name=f"daemon-{self._session_id}")
        logger.info("daemon_started", session_id=self._session_id)

    async def stop(self) -> None:
        """Stop the daemon gracefully.

        Flushes any remaining messages in the buffer before shutting down.
        """
        self._running = False
        self._buffer_event.set()  # Wake up the loop so it can exit

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except TimeoutError:
                logger.warning("daemon_stop_timeout", session_id=self._session_id)
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        # Close the direct Elasticsearch client
        await self._es.close()

        logger.info("daemon_stopped", session_id=self._session_id)

    async def flush(self) -> SemanticAtom | None:
        """Force-compress whatever is in the buffer, regardless of threshold.

        Returns:
            The created SemanticAtom, or None if the buffer is empty.
        """
        if not self._message_buffer:
            return None

        return await self._compress_and_store()

    async def _run_loop(self) -> None:
        """Main daemon loop — waits for buffer threshold, then compresses."""
        logger.info("daemon_loop_started", session_id=self._session_id)

        while self._running:
            # Wait for the buffer to fill up
            await self._buffer_event.wait()
            self._buffer_event.clear()

            if not self._running:
                break

            # Process if we have enough messages
            while len(self._message_buffer) >= self._threshold:
                try:
                    await self._compress_and_store()
                except Exception as e:
                    logger.error("daemon_compression_cycle_failed", error=str(e))
                    # Don't crash the loop on individual failures
                    await asyncio.sleep(1.0)

        # Flush remaining messages on shutdown
        if self._message_buffer:
            logger.info("daemon_flushing_on_shutdown", remaining=len(self._message_buffer))
            try:
                await self._compress_and_store()
            except Exception as e:
                logger.error("daemon_flush_failed", error=str(e))

    async def _compress_and_store(self) -> SemanticAtom:
        """Take messages from the buffer, compress them, and store the atom.

        Returns:
            The created SemanticAtom.

        Raises:
            CompressionError: If LLM compression fails.
            StorageError: If writing to Elastic fails.
        """
        # Take messages from the buffer
        messages_to_compress = self._message_buffer[: self._threshold]
        self._message_buffer = self._message_buffer[self._threshold :]

        logger.info(
            "compressing_messages",
            count=len(messages_to_compress),
            remaining_in_buffer=len(self._message_buffer),
        )

        # Compress with the LLM
        atom = await self._compress_messages(messages_to_compress)

        # Store in Elastic via MCP
        await self._store_atom(atom)

        return atom

    async def _compress_messages(self, messages: list[Message]) -> SemanticAtom:
        """Use the LLM to compress messages into a semantic atom.

        Args:
            messages: The messages to compress.

        Returns:
            A SemanticAtom containing the summary and metadata.

        Raises:
            CompressionError: If the LLM fails to produce a valid summary.
        """
        # Format messages for the compression prompt
        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == Role.USER else "Assistant"
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            formatted_lines.append(f"[{timestamp}] {role_label}: {msg.content}")

        conversation_text = "\n".join(formatted_lines)
        prompt = f"Compress the following conversation chunk:\n\n{conversation_text}"

        try:
            response = await self._client.aio.models.generate_content(
                model=self._config.router_model,  # Use the fast model for compression
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=COMPRESSION_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise CompressionError("LLM returned empty compression response")

            # Parse the JSON response
            import json

            cleaned = response.text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            data = json.loads(cleaned)

            atom = SemanticAtom(
                summary=data.get("summary", ""),
                keywords=data.get("keywords", []),
                source_message_ids=[msg.id for msg in messages],
                source_message_count=len(messages),
                session_id=self._session_id,
                time_range_start=messages[0].timestamp,
                time_range_end=messages[-1].timestamp,
            )

            logger.info(
                "messages_compressed",
                atom_id=atom.id,
                summary_length=len(atom.summary),
                keyword_count=len(atom.keywords),
            )

            return atom

        except CompressionError:
            raise
        except Exception as e:
            raise CompressionError(f"Failed to compress messages: {e}") from e

    async def _store_atom(self, atom: SemanticAtom) -> None:
        """Write a semantic atom to Elasticsearch using the direct async client.

        Bypasses the MCP server (which is read-only) and writes directly
        to the Elasticsearch cluster.

        Args:
            atom: The semantic atom to store.

        Raises:
            StorageError: If the indexing operation fails.
        """
        try:
            doc = atom.to_elastic_document()
            result = await self._es.index(
                index=self._config.elastic_index,
                id=atom.id,
                document=doc,
            )

            logger.info(
                "atom_stored",
                atom_id=atom.id,
                index=self._config.elastic_index,
                es_result=result.get("result", "unknown"),
            )

        except Exception as e:
            raise StorageError(f"Failed to store atom {atom.id}: {e}") from e

