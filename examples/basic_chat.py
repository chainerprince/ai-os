#!/usr/bin/env python3
"""
Basic Chat Example — Agentic Memory OS

A simple CLI script demonstrating how to use the SDK to chat with
the Agentic Memory OS. Shows the full session lifecycle:
1. Create config from environment
2. Start a session (connects MCP, starts daemon)
3. Chat in a loop
4. Graceful shutdown on exit

Usage:
    # Make sure .env is configured (see .env.example)
    uv run python examples/basic_chat.py
"""

from __future__ import annotations

import asyncio
import sys

from agent_os import AgentOSConfig, Session


async def main() -> None:
    """Run an interactive chat session with Agentic Memory OS."""

    print("=" * 60)
    print("  🧠 Agentic Memory OS — Basic Chat")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    print()

    # Load configuration from environment / .env file
    try:
        config = AgentOSConfig()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("   Make sure you have a .env file with GEMINI_API_KEY set.")
        print("   See .env.example for the template.")
        sys.exit(1)

    # Start a session — this connects to Elastic MCP and starts the daemon
    async with Session(config) as session:
        print(f"✅ Session started (ID: {session.session_id[:8]}...)")
        print(f"   Kernel model:  {config.kernel_model}")
        print(f"   Router model:  {config.router_model}")
        print(f"   Memory index:  {config.elastic_index}")
        print()

        turn = 0
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            turn += 1
            try:
                response = await session.chat(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

        print(f"Session ended after {turn} turns ({session.message_count} messages).")
        print("Flushing memory daemon...")

    print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
