"""
Agentic Memory OS — An OS-inspired Memory Management Unit for LLMs.

Solves the "Lost in the Middle" hallucination problem by routing only
relevant semantic context into the LLM's context window.
"""

__version__ = "0.1.0"

from agent_os.config import AgentOSConfig
from agent_os.session import Session

__all__ = [
    "AgentOSConfig",
    "Session",
    "__version__",
]
