# 🧠 Agentic Memory OS

> An OS-inspired Memory Management Unit for LLMs — solving the "Lost in the Middle" problem.

## The Problem

LLMs suffer from **context window bloat**. As conversations grow, stuffing the entire chat history into the context leads to hallucinations, lost information, and wasted tokens. Research shows LLMs struggle most with information buried in the *middle* of long contexts.

## The Solution

**Agentic Memory OS** treats LLM memory like a computer operating system:

| OS Concept | SDK Component | What It Does |
|---|---|---|
| **CPU** | `SystemKernel` | Runs the main LLM (Gemini 3.5 Flash) with only relevant context injected |
| **MMU** | `RouterMMU` | A fast router (Gemini 3.1 Flash-Lite) that decides *what* context to fetch |
| **Swap Space** | Elasticsearch via MCP | Persistent memory store with hybrid search (semantic + keyword) |
| **Background Daemon** | `MemoryDaemon` | Compresses old messages into "semantic atoms" and writes them to storage |

Instead of passing everything to the LLM, the router fetches only what's relevant — like an OS paging memory in and out.

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd ai-os
uv sync --all-extras

# Configure
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and Elasticsearch details

# Run the demo
uv run python examples/basic_chat.py
```

## Architecture

```
User Prompt
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  RouterMMU  │────▶│  Elastic MCP     │
│ (Flash-Lite)│◀────│  (Hybrid Search) │
└──────┬──────┘     └──────────────────┘
       │ relevant context
       ▼
┌─────────────┐
│ SystemKernel│────▶ Response to User
│  (3.5 Flash)│
└─────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────────┐
│ MemoryDaemon │───▶│  Elastic MCP     │
│ (background) │    │  (Write Atoms)   │
└──────────────┘    └──────────────────┘
```

## Tech Stack

- **Python 3.12+** with `uv` for package management
- **Gemini 3.5 Flash** — main LLM (the "CPU")
- **Gemini 3.1 Flash-Lite** — routing LLM (the "MMU")
- **Elasticsearch** — vector + keyword storage (the "Swap Space")
- **Model Context Protocol (MCP)** — standardized tool access to Elasticsearch
- **Pydantic** — data validation and configuration

## Project Structure

```
src/agent_os/
├── config.py       # Environment-based configuration
├── kernel.py       # SystemKernel — main LLM interaction
├── router.py       # RouterMMU — context routing decisions
├── daemon.py       # MemoryDaemon — async memory compressor
├── session.py      # Session — user-facing orchestrator
├── mcp_client.py   # Elastic MCP server connection
├── models.py       # Pydantic data models
├── exceptions.py   # Custom exceptions
└── logging.py      # Structured logging
```

## License

MIT
