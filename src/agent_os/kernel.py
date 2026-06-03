"""
SystemKernel — The CPU of Agentic Memory OS.

Handles the main LLM interaction with the user. Receives pre-fetched
context from the RouterMMU and generates responses. Does NOT manage
its own conversation history — that's the Session's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

from agent_os.exceptions import GenerationError
from agent_os.logging import get_logger
from agent_os.models import Message, RetrievedContext, Role

if TYPE_CHECKING:
    from agent_os.config import AgentOSConfig

logger = get_logger("kernel")

# System prompt that instructs the kernel how to use injected context
KERNEL_SYSTEM_PROMPT = """\
You are a helpful AI assistant powered by the Agentic Memory OS.

When you receive context from previous conversations (marked with [Memory from ...]),
use it naturally to provide informed, coherent responses. Do NOT explicitly mention
that you are "retrieving memories" — just use the information as if you remember it.

If no historical context is provided, respond normally based on the current conversation.
Be concise, accurate, and helpful."""


class SystemKernel:
    """The CPU — main LLM interaction layer.

    Takes a user prompt + optional injected context and produces a response.
    Uses the heavier/smarter model (Gemini 3.5 Flash by default).

    The kernel is stateless with respect to conversation history. The Session
    maintains the message list and feeds recent messages to the kernel each turn.
    """

    def __init__(self, config: AgentOSConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.kernel_model

        logger.info("kernel_initialized", model=self._model)

    async def generate(
        self,
        user_message: str,
        recent_messages: list[Message] | None = None,
        retrieved_context: RetrievedContext | None = None,
    ) -> str:
        """Generate a response to the user's message.

        Args:
            user_message: The current user prompt.
            recent_messages: Recent conversation messages for short-term context.
            retrieved_context: Historical context fetched by the RouterMMU.

        Returns:
            The generated response text.

        Raises:
            GenerationError: If the LLM fails to generate a response.
        """
        # Build the contents list for the API call
        contents = self._build_contents(user_message, recent_messages, retrieved_context)

        logger.info(
            "kernel_generating",
            model=self._model,
            has_context=retrieved_context is not None and not retrieved_context.is_empty,
            recent_message_count=len(recent_messages) if recent_messages else 0,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=KERNEL_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )

            if not response.text:
                raise GenerationError("LLM returned an empty response")

            logger.info(
                "kernel_response_generated",
                response_length=len(response.text),
            )

            return response.text

        except GenerationError:
            raise
        except Exception as e:
            logger.error("kernel_generation_failed", error=str(e))
            raise GenerationError(f"Failed to generate response: {e}") from e

    def _build_contents(
        self,
        user_message: str,
        recent_messages: list[Message] | None,
        retrieved_context: RetrievedContext | None,
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        """Build the contents array for the Gemini API call.

        Combines historical context, recent messages, and the current prompt
        into the format expected by the google-genai SDK.
        """
        contents: list[dict[str, str | list[dict[str, str]]]] = []

        # Inject retrieved historical context as a system-like preamble
        if retrieved_context and not retrieved_context.is_empty:
            context_text = (
                "Here is relevant context from previous conversations:\n\n"
                f"{retrieved_context.combined_summary}\n\n"
                "Use this context to inform your response."
            )
            contents.append({
                "role": "user",
                "parts": [{"text": context_text}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I'll use this context to provide an informed response."}],
            })

        # Add recent conversation messages for short-term memory
        if recent_messages:
            for msg in recent_messages:
                role = "model" if msg.role == Role.ASSISTANT else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })

        # Add the current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}],
        })

        return contents
