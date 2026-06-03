"""
RouterMMU — The Memory Management Unit of Agentic Memory OS.

Intercepts user prompts and decides whether historical context is needed.
If so, it formulates a search query and retrieves relevant semantic atoms
from the Elastic MCP server using hybrid search (semantic + BM25).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from google import genai

from agent_os.exceptions import RoutingDecisionError
from agent_os.logging import get_logger
from agent_os.models import Message, RetrievedContext, Role, RoutingDecision, SemanticAtom

if TYPE_CHECKING:
    from agent_os.config import AgentOSConfig
    from agent_os.mcp_client import ElasticMCPClient

logger = get_logger("router")

# System prompt for the router — instructs it to make routing decisions
ROUTER_SYSTEM_PROMPT = """\
You are a routing agent inside a Memory Management Unit (MMU). Your ONLY job is to
decide whether a user's message requires historical context from previous conversations.

Analyze the user's message and the recent conversation. Respond with a JSON object:

{
  "needs_context": true/false,
  "search_query": "optimized search query for semantic + keyword search" or null,
  "reasoning": "brief explanation of your decision"
}

Guidelines:
- Return needs_context=true when the user refers to past conversations, previous topics,
  things they mentioned before, or asks follow-up questions that need historical context.
- Return needs_context=false for greetings, simple factual questions, or when the current
  conversation provides sufficient context.
- When needs_context=true, formulate search_query as a concise, keyword-rich query
  optimized for hybrid search (combine semantic meaning + key terms).
- ALWAYS respond with valid JSON only. No other text."""


class RouterMMU:
    """The MMU — context routing and retrieval layer.

    Uses a fast, cheap model (Gemini 3.1 Flash-Lite by default) to analyze
    each user prompt and decide:
    1. Does this prompt need historical context?
    2. If yes, what should we search for?

    When context is needed, it queries the Elastic MCP server using
    hybrid search and returns the results for injection into the Kernel.
    """

    def __init__(self, config: AgentOSConfig, mcp_client: ElasticMCPClient) -> None:
        self._config = config
        self._mcp_client = mcp_client
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.router_model

        logger.info("router_initialized", model=self._model)

    async def route(
        self,
        user_message: str,
        recent_messages: list[Message] | None = None,
    ) -> RetrievedContext:
        """Route a user message: decide if context is needed and fetch it.

        This is the main entry point. It:
        1. Asks the routing LLM if context is needed
        2. If yes, queries the Elastic MCP server
        3. Returns the retrieved context (empty if not needed)

        Args:
            user_message: The current user prompt.
            recent_messages: Recent messages for routing context.

        Returns:
            RetrievedContext — either populated with atoms or empty.
        """
        # Step 1: Get routing decision from the fast LLM
        decision = await self._make_routing_decision(user_message, recent_messages)

        logger.info(
            "routing_decision",
            needs_context=decision.needs_context,
            search_query=decision.search_query,
            reasoning=decision.reasoning,
        )

        if not decision.needs_context or not decision.search_query:
            return RetrievedContext()

        # Step 2: Fetch context from Elastic MCP
        return await self._fetch_context(decision.search_query)

    async def _make_routing_decision(
        self,
        user_message: str,
        recent_messages: list[Message] | None = None,
    ) -> RoutingDecision:
        """Ask the routing LLM whether context is needed.

        Args:
            user_message: The user's current message.
            recent_messages: Recent conversation for context.

        Returns:
            A RoutingDecision parsed from the LLM's JSON response.

        Raises:
            RoutingDecisionError: If the LLM response cannot be parsed.
        """
        # Build a summary of recent conversation for the router
        conversation_context = ""
        if recent_messages:
            lines = []
            for msg in recent_messages[-5:]:  # Only last 5 messages for the router
                role_label = "User" if msg.role == Role.USER else "Assistant"
                lines.append(f"{role_label}: {msg.content[:200]}")
            conversation_context = (
                "Recent conversation:\n" + "\n".join(lines) + "\n\n"
            )

        prompt = f"{conversation_context}Current user message: {user_message}"

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=ROUTER_SYSTEM_PROMPT,
                    temperature=0.1,  # Low temperature for deterministic routing
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                logger.warning("router_empty_response")
                return RoutingDecision(needs_context=False, reasoning="Empty router response")

            return self._parse_decision(response.text)

        except Exception as e:
            logger.warning("routing_decision_failed", error=str(e))
            # Fail open: if routing fails, skip context retrieval
            return RoutingDecision(
                needs_context=False,
                reasoning=f"Routing failed, skipping context: {e}",
            )

    def _parse_decision(self, response_text: str) -> RoutingDecision:
        """Parse the routing LLM's JSON response into a RoutingDecision.

        Args:
            response_text: Raw text from the routing LLM.

        Returns:
            A validated RoutingDecision.

        Raises:
            RoutingDecisionError: If the response is not valid JSON or
                                   missing required fields.
        """
        # Strip any markdown code fence wrapping
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RoutingDecisionError(
                f"Router returned invalid JSON: {response_text[:200]}"
            ) from e

        try:
            return RoutingDecision(**data)
        except Exception as e:
            raise RoutingDecisionError(
                f"Router JSON missing required fields: {e}"
            ) from e

    async def _fetch_context(self, search_query: str) -> RetrievedContext:
        """Fetch relevant context from the Elastic MCP server.

        Args:
            search_query: The optimized search query from the routing decision.

        Returns:
            RetrievedContext populated with matching semantic atoms.
        """
        logger.info("fetching_context", query=search_query)

        try:
            raw_results = await self._mcp_client.search(query=search_query)

            # Parse results into SemanticAtom objects
            atoms = []
            for result in raw_results:
                try:
                    if "_raw" in result:
                        # MCP server returns text-format results — parse them
                        atom = self._parse_text_result(result["_raw"])
                        if atom:
                            atoms.append(atom)
                    else:
                        # Structured JSON result — try direct parsing
                        atom = SemanticAtom(**result)
                        atoms.append(atom)
                except Exception:
                    logger.debug(
                        "skipped_non_atom_result",
                        result_keys=list(result.keys()) if isinstance(result, dict) else "non-dict",
                    )

            context = RetrievedContext(atoms=atoms, raw_results=raw_results)

            logger.info(
                "context_retrieved",
                atom_count=len(atoms),
                raw_result_count=len(raw_results),
            )

            return context

        except Exception as e:
            logger.warning("context_fetch_failed", error=str(e))
            # Fail open: return empty context rather than crashing
            return RetrievedContext()

    @staticmethod
    def _parse_text_result(text: str) -> SemanticAtom | None:
        """Parse a text-format result from the Elastic MCP server into a SemanticAtom.

        The MCP server returns results like:
            summary (highlighted): User discussed building...
            keywords: ["hackathon","memory"]
            session_id: "test-session-001"

        Args:
            text: The raw text result from the MCP server.

        Returns:
            A SemanticAtom if parsing succeeds, None otherwise.
        """
        import re

        # Strip HTML highlight tags
        clean_text = re.sub(r"</?em>", "", text)

        # Skip the "Total results:" header line
        lines = clean_text.strip().split("\n")

        summary = ""
        keywords: list[str] = []
        session_id = ""

        for line in lines:
            line = line.strip()
            if line.startswith("summary"):
                # "summary (highlighted): actual summary text"
                # or "summary: actual summary text"
                summary = re.sub(r"^summary\s*(\(highlighted\))?\s*:\s*", "", line)
            elif line.startswith("keywords:"):
                kw_str = line.split(":", 1)[1].strip()
                try:
                    keywords = json.loads(kw_str)
                except (json.JSONDecodeError, TypeError):
                    keywords = [k.strip().strip('"') for k in kw_str.strip("[]").split(",")]
            elif line.startswith("session_id:"):
                session_id = line.split(":", 1)[1].strip().strip('"')

        if not summary:
            return None

        return SemanticAtom(
            summary=summary,
            keywords=keywords,
            session_id=session_id or "unknown",
        )

