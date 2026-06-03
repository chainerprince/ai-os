# Project Context: "Agentic Memory OS" (Hackathon Build)

I am building a multi-agent Python SDK for an AI hackathon. The goal is to solve the "Lost in the Middle" hallucination problem and reduce context-window bloat by building a Memory Management Unit (MMU) for LLMs, modeled after a computer operating system.

Instead of passing the entire chat history into the LLM context window every time, this SDK uses small routing agents to fetch only the relevant semantic context from an Elastic database via the Model Context Protocol (MCP) before waking the main LLM.

# Architecture & Tech Stack

- **Language:** Python 3.12+
- **Package Management:** `uv` with `pyproject.toml`
- **The CPU (Main LLM):** Gemini 3.5 Flash (`gemini-3.5-flash`) via `google-genai >= 2.7.0`
- **The MMU (Router):** Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) — fast, cheap routing decisions
- **Swap Space (Storage):** Elasticsearch, accessed via the official Elastic MCP Server (`@elastic/mcp-server-elasticsearch`)
- **MCP Client:** Official MCP Python SDK (`mcp[cli]`) for connecting to the Elastic MCP server over stdio
- **Async Runtime:** `asyncio` (no external server framework — the SDK is self-contained)

# Core SDK Components

1. **`SystemKernel` (The CPU):** A class that handles the main interaction with the user using Gemini 3.5 Flash. It accepts injected context but does NOT manage its own history.
2. **`RouterMMU` (The MMU):** A class using Gemini 3.1 Flash-Lite. It intercepts user prompts, determines if historical context is needed, and if so, executes a tool call to the Elastic MCP server using hybrid search (semantic + BM25 keyword).
3. **`MemoryDaemon` (The Background Worker):** An asyncio background task. It monitors the active session. When 10 messages accumulate, it compresses them into a "semantic atom" (summary + metadata) and writes it to the Elastic MCP server.
4. **`Session` (The Orchestrator):** The user-facing entry point that ties together the Kernel, Router, and Daemon for a single conversation.

# Supporting Modules

- **`config.py`** — Pydantic `BaseSettings` for environment-based configuration (API keys, Elastic URL, model names)
- **`models.py`** — Pydantic data models (`SemanticAtom`, `Message`, `RoutingDecision`, etc.)
- **`mcp_client.py`** — Wrapper around the MCP Python SDK to manage the Elastic MCP server connection lifecycle
- **`exceptions.py`** — Custom exception hierarchy for clear error handling
- **`logging.py`** — Structured logging configuration

# Directory Structure

```text
ai-os/
├── pyproject.toml              # All metadata, deps, tool config
├── uv.lock                     # Reproducible installs (generated)
├── .python-version             # "3.12"
├── .env.example                # Template: GEMINI_API_KEY, ELASTIC_URL, etc.
├── .gitignore
├── README.md
│
├── src/
│   └── agent_os/
│       ├── __init__.py         # Package init, version
│       ├── config.py           # Pydantic BaseSettings
│       ├── kernel.py           # SystemKernel
│       ├── router.py           # RouterMMU
│       ├── daemon.py           # MemoryDaemon
│       ├── session.py          # Session orchestrator
│       ├── mcp_client.py       # Elastic MCP client wrapper
│       ├── models.py           # Pydantic data models
│       ├── exceptions.py       # Custom exceptions
│       └── logging.py          # Logging setup
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_kernel.py
│   ├── test_router.py
│   └── test_daemon.py
│
└── examples/
    └── basic_chat.py           # CLI demo
```