"""Adapter that plugs a *Trainable* ART model into the HUD MCPAgent stack.

This extends GenericOpenAIChatAgent to collect messages_and_choices during
execution for ART training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent

if TYPE_CHECKING:
    import art

    from hud.clients import AgentMCPClient
    from hud.types import AgentResponse

logger = logging.getLogger(__name__)


system_prompt = (
    "You are an MCP (Model Context Protocol) agent.\n\n"
    "Use MCP tools through the server to complete your task.\n\n"
    "You have a total of {MAX_STEPS} steps."
    "You are an excellent 2048 player. Always choose the move most likely to combine tiles and reach higher numbers."  # noqa: E501
    "Only provide a maximum of one tool call at a time."
)


class ArtHUDAgent(GenericOpenAIChatAgent):
    """Use an ART *TrainableModel* as the LLM behind a HUD `MCPAgent`.

    This agent collects messages_and_choices during execution for ART training.
    """

    def __init__(
        self, art_model: art.Model, mcp_client: AgentMCPClient, **agent_kwargs: Any
    ) -> None:
        # Use ART's openai_client() method to get proper timeouts and patching
        openai_client = art_model.openai_client()

        super().__init__(
            mcp_client=mcp_client,
            openai_client=openai_client,
            model_name=art_model.get_inference_name(),
            **agent_kwargs,
        )
        self.custom_system_prompt = system_prompt

        self.art_model = art_model
        self.messages_and_choices: list[Any] = []  # Collect for ART training

        logger.info(
            "ArtHUDAgent initialised with model '%s' (project=%s)",
            art_model.name,
            getattr(art_model, "project", "unknown"),
        )

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[Any]:
        """Create initial messages and store them for ART."""
        messages = await super().create_initial_messages(prompt, initial_screenshot)
        # Store initial messages as dicts for ART
        self.messages_and_choices.extend(messages)
        return messages

    @hud.instrument(
        span_type="agent",
        record_args=False,  # Messages can be large
        record_result=True,
    )
    async def get_model_response(self, messages: list[Any]) -> AgentResponse:
        """Get model response and store the Choice for ART."""
        # Call parent's get_model_response
        result = await super().get_model_response(messages)

        # Extract and store the Choice from the raw response
        if result.raw and hasattr(result.raw, "choices") and result.raw.choices:
            choice = result.raw.choices[0]
            # Ensure the message has content (required for ART tokenization)
            if choice.message and choice.message.content is None:
                choice.message.content = ""
            self.messages_and_choices.append(choice)

        return result

    async def format_tool_results(
        self, tool_calls: list[Any], tool_results: list[Any]
    ) -> list[Any]:
        """Format tool results and store them for ART."""
        tool_messages = await super().format_tool_results(tool_calls, tool_results)
        # Store tool messages for ART
        self.messages_and_choices.extend(tool_messages)
        return tool_messages
