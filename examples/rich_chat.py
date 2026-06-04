#!/usr/bin/env python3
"""
Rich UI Chat Example — Agentic Memory OS

A beautiful terminal interface demonstrating the Agentic Memory OS metaphor.
It displays the chat and a "System Monitor" showing RAM usage, Swap atoms,
Page Faults, and Token/Cost savings after every turn.

Usage:
    uv run python examples/rich_chat.py
"""

import asyncio
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_os import AgentOSConfig, Session


async def main() -> None:
    console = Console()

    # Suppress verbose debug logs to keep the UI clean
    import logging

    logging.getLogger().setLevel(logging.CRITICAL)

    console.print(
        Panel(
            "[bold cyan]🧠 Agentic Memory OS[/] — Hackathon Demo\n"
            "[dim]Type 'quit' or 'exit' to end the session.[/]",
            title="Welcome",
            border_style="cyan",
        )
    )

    try:
        config = AgentOSConfig()
    except Exception as e:
        console.print(f"[red]❌ Configuration error:[/] {e}")
        sys.exit(1)

    # Force threshold lower so the judges can see compression happen quickly
    config.compression_threshold = 3

    async with Session(config) as session:
        console.print(f"[green]✅ Session started (ID: {session.session_id[:8]}...)[/]")
        console.print(f"[dim]Kernel: {config.kernel_model} | Router: {config.router_model}[/]\n")

        turn = 0
        while True:
            try:
                # Use standard input
                user_input = console.input("[bold cyan]You:[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not user_input.strip():
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            turn += 1

            with console.status("[bold magenta]Processing (MMU -> Kernel)...[/]", spinner="dots"):
                try:
                    response = await session.chat(user_input)
                except Exception as e:
                    console.print(f"[red]❌ Error:[/] {e}")
                    continue

            # Print Assistant Response
            console.print(f"\n[bold magenta]Assistant:[/] {response}\n")

            # Print System Monitor Metrics
            metrics = session.last_turn_metrics

            table = Table(box=box.ROUNDED, border_style="green", show_lines=True)
            table.add_column("OS Component", style="bold white")
            table.add_column("Metric", justify="right")
            table.add_column("Value", justify="left")

            # CPU (Kernel)
            table.add_row("CPU (Kernel)", "Latency", f"{metrics['latency_sec']:.2f}s")

            # MMU (Router)
            pf_color = "red bold" if metrics["page_fault"] else "dim"
            pf_text = f"[{pf_color}]YES[/]" if metrics["page_fault"] else f"[{pf_color}]NO[/]"
            table.add_row("MMU (Router)", "Page Fault (Context)", pf_text)

            # RAM (Active Context Window)
            ram_color = (
                "yellow" if session.message_count >= config.compression_threshold else "green"
            )
            table.add_row(
                "RAM (Session)", "Active Messages", f"[{ram_color}]{session.message_count}[/]"
            )

            # Daemon (Background Worker)
            table.add_row("Daemon", "Buffer Size", str(metrics["daemon_buffer"]))

            # Swap (Elasticsearch)
            table.add_row(
                "Swap (Disk)", "Semantic Atoms", f"[magenta]{session.total_atoms_swapped}[/]"
            )
            table.add_row(
                "Swap (Disk)", "Tokens Saved", f"[bold green]{session.total_tokens_saved}[/]"
            )

            console.print(table)
            console.print()

        console.print(f"[dim]Session ended. Total turns: {turn}[/]")


if __name__ == "__main__":
    asyncio.run(main())
