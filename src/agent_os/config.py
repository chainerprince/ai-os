"""
Configuration management for Agentic Memory OS.

Uses Pydantic BaseSettings to load configuration from environment variables
and .env files. All credentials and model settings are centralized here.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AgentOSConfig(BaseSettings):
    """Central configuration for all Agentic Memory OS components.

    Loads values from environment variables and .env files.
    Environment variables take precedence over .env file values.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Gemini API ---
    gemini_api_key: str = Field(description="Google Gemini API key")

    # --- Model selection ---
    kernel_model: str = Field(
        default="gemini-3.5-flash",
        description="Model ID for the SystemKernel (main LLM)",
    )
    router_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Model ID for the RouterMMU (routing LLM)",
    )

    # --- Elasticsearch MCP Server ---
    elastic_mcp_server_cmd: str = Field(
        default="npx",
        description="Command to launch the Elastic MCP server",
    )
    elastic_mcp_server_args: str = Field(
        default="-y,@elastic/mcp-server-elasticsearch",
        description="Comma-separated args for the Elastic MCP server command",
    )
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        description="Elasticsearch connection URL",
    )
    elasticsearch_api_key: str = Field(
        default="",
        description="Elasticsearch API key for authentication",
    )

    # --- Memory Daemon ---
    compression_threshold: int = Field(
        default=10,
        description="Number of messages before the daemon compresses them into a semantic atom",
    )
    elastic_index: str = Field(
        default="agent_os_memory",
        description="Elasticsearch index name for storing semantic atoms",
    )

    @property
    def elastic_mcp_args_list(self) -> list[str]:
        """Return MCP server args as a list, split on commas."""
        return [arg.strip() for arg in self.elastic_mcp_server_args.split(",") if arg.strip()]

    @property
    def elastic_env(self) -> dict[str, str]:
        """Return environment variables to pass to the Elastic MCP server process."""
        env = {
            "ELASTICSEARCH_URL": self.elasticsearch_url,
            "ES_URL": self.elasticsearch_url,
            "OTEL_SDK_DISABLED": "true",
            "OTEL_LOG_LEVEL": "warn",
        }
        if self.elasticsearch_api_key:
            env["ELASTICSEARCH_API_KEY"] = self.elasticsearch_api_key
            env["ES_API_KEY"] = self.elasticsearch_api_key
        return env
