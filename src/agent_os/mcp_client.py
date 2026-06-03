"""
MCP client wrapper for connecting to the Elastic MCP server.

Manages the lifecycle of the MCP server subprocess and provides
high-level methods for searching and indexing semantic atoms.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_os.exceptions import MCPConnectionError, MCPToolCallError
from agent_os.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from agent_os.config import AgentOSConfig

logger = get_logger("mcp_client")


class ElasticMCPClient:
    """Wrapper around the MCP Python SDK for Elastic MCP server communication.

    Manages the stdio transport to the Elastic MCP server subprocess and
    exposes high-level methods for search and indexing operations.

    Usage:
        async with ElasticMCPClient(config) as client:
            results = await client.search("what did the user say about Python?")
            await client.index_document(index="memory", doc={...})
    """

    def __init__(self, config: AgentOSConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._available_tools: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Check if the MCP client has an active session."""
        return self._session is not None

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[ElasticMCPClient, None]:
        """Open a connection to the Elastic MCP server.

        Starts the MCP server as a subprocess, initializes the session,
        and discovers available tools.

        Yields:
            self — the connected client instance.

        Raises:
            MCPConnectionError: If the server fails to start or initialize.
        """
        server_params = StdioServerParameters(
            command=self._config.elastic_mcp_server_cmd,
            args=self._config.elastic_mcp_args_list,
            env={**os.environ, **self._config.elastic_env},
        )

        logger.info(
            "connecting_to_elastic_mcp",
            command=self._config.elastic_mcp_server_cmd,
            args=self._config.elastic_mcp_args_list,
        )

        try:
            async with (
                stdio_client(server_params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                self._session = session

                # Initialize the MCP session and discover tools
                await session.initialize()
                tools_response = await session.list_tools()
                self._available_tools = [
                    {"name": tool.name, "description": tool.description}
                    for tool in tools_response.tools
                ]

                logger.info(
                    "mcp_connected",
                    tools_discovered=len(self._available_tools),
                    tool_names=[t["name"] for t in self._available_tools],
                )

                yield self

        except Exception as e:
            logger.error("mcp_connection_failed", error=str(e))
            raise MCPConnectionError(f"Failed to connect to Elastic MCP server: {e}") from e
        finally:
            self._session = None
            self._available_tools = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool on the Elastic server.

        Args:
            tool_name: Name of the MCP tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's response content.

        Raises:
            MCPConnectionError: If not connected.
            MCPToolCallError: If the tool call fails.
        """
        if self._session is None:
            raise MCPConnectionError("Not connected. Use `async with client.connect()` first.")

        logger.debug("mcp_tool_call", tool=tool_name, arguments=arguments)

        try:
            result = await self._session.call_tool(tool_name, arguments)

            if result.isError:
                error_msg = str(result.content) if result.content else "Unknown error"
                raise MCPToolCallError(tool_name, error_msg)

            # Extract text content from the result
            content_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    content_parts.append(block.text)

            combined = "\n".join(content_parts)
            logger.debug("mcp_tool_result", tool=tool_name, result_length=len(combined))

            # Try to parse as JSON, fall back to raw text
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, TypeError):
                return combined

        except MCPToolCallError:
            raise
        except Exception as e:
            logger.error("mcp_tool_call_failed", tool=tool_name, error=str(e))
            raise MCPToolCallError(tool_name, str(e)) from e

    async def search(
        self,
        query: str,
        index: str | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for semantic atoms in Elasticsearch using hybrid search.

        Args:
            query: The natural-language search query.
            index: Elasticsearch index to search (defaults to config value).
            max_results: Maximum number of results to return.

        Returns:
            List of matching documents from Elasticsearch.
        """
        target_index = index or self._config.elastic_index

        # Use the Elastic MCP search tool — expects `queryBody` as an object
        search_body = {
            "index": target_index,
            "queryBody": {
                "query": {
                    "bool": {
                        "should": [
                            # BM25 keyword search on summary + keywords
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["summary", "keywords"],
                                    "type": "best_fields",
                                }
                            },
                        ]
                    }
                },
                "size": max_results,
            },
        }

        result = await self.call_tool("search", search_body)

        # The MCP server may return structured JSON or plain text
        if isinstance(result, dict) and "hits" in result:
            return [hit.get("_source", hit) for hit in result["hits"].get("hits", [])]
        if isinstance(result, list):
            return result

        # Handle text responses like "Total results: 0, showing 0 from position 0"
        # or error messages like "Search failed: index_not_found_exception..."
        if isinstance(result, str):
            is_empty = (
                result.startswith("Total results: 0")
                or result.startswith("Error")
                or result.startswith("Search failed")
            )
            if is_empty:
                return []
            # Non-empty text result — wrap it
            return [{"_raw": result}] if result.strip() else []

        return [result] if result else []

    async def index_document(
        self,
        document: dict[str, Any],
        index: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Index a document (e.g., a semantic atom) into Elasticsearch.

        Args:
            document: The document body to index.
            index: Target Elasticsearch index (defaults to config value).
            doc_id: Optional document ID. Auto-generated if not provided.

        Returns:
            The indexing response from Elasticsearch.
        """
        target_index = index or self._config.elastic_index

        arguments: dict[str, Any] = {
            "index": target_index,
            "body": json.dumps(document),
        }
        if doc_id:
            arguments["id"] = doc_id

        return await self.call_tool("index", arguments)

    @property
    def available_tools(self) -> list[dict[str, Any]]:
        """List of tools discovered on the MCP server."""
        return list(self._available_tools)
