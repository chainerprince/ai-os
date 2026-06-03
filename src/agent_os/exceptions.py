"""
Custom exception hierarchy for Agentic Memory OS.

Provides specific error types for each component, making it easy to
catch and handle failures at the appropriate level.
"""

from __future__ import annotations


class AgentOSError(Exception):
    """Base exception for all Agentic Memory OS errors."""


# --- Configuration ---


class ConfigError(AgentOSError):
    """Raised when configuration is missing or invalid."""


# --- MCP Client ---


class MCPError(AgentOSError):
    """Base exception for MCP-related errors."""


class MCPConnectionError(MCPError):
    """Raised when the MCP client cannot connect to the server."""


class MCPToolCallError(MCPError):
    """Raised when an MCP tool call fails."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"MCP tool '{tool_name}' failed: {message}")


# --- Kernel ---


class KernelError(AgentOSError):
    """Base exception for SystemKernel errors."""


class GenerationError(KernelError):
    """Raised when the LLM fails to generate a response."""


# --- Router ---


class RouterError(AgentOSError):
    """Base exception for RouterMMU errors."""


class RoutingDecisionError(RouterError):
    """Raised when the router cannot parse a routing decision from the LLM response."""


# --- Daemon ---


class DaemonError(AgentOSError):
    """Base exception for MemoryDaemon errors."""


class CompressionError(DaemonError):
    """Raised when the daemon fails to compress messages into a semantic atom."""


class StorageError(DaemonError):
    """Raised when the daemon fails to write a semantic atom to storage."""


# --- Session ---


class SessionError(AgentOSError):
    """Base exception for Session errors."""


class SessionNotStartedError(SessionError):
    """Raised when an operation is attempted on a session that hasn't been started."""

    def __init__(self) -> None:
        super().__init__("Session has not been started. Call `await session.start()` first.")
