# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from anthropic import AsyncAnthropic

from hud.adapters import Adapter
from hud.adapters.common.types import CLA

# Update import to current API; if this script is legacy, keep it optional
try:
    from hud.agent import MCPAgent as Agent  # type: ignore[assignment]
except Exception:  # pragma: no cover - optional example script
    from hud.agent import MCPAgent as Agent  # fallback
from hud.settings import settings

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaImageBlockParam,
        BetaMessageParam,
        BetaTextBlockParam,
    )

    from hud.env.environment import Observation

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_MESSAGE_MEMORY = 20


def generate_system_prompt(game_name: str) -> str:
    """Generate the system prompt for the AI agent.

    Args:
        game_name: Name of the game being played

    Returns:
        str: The system prompt for the AI agent
    """
    return """You are a specialized AI assistant designed to play Pokémon games via screenshot analysis and text instructions. Your task is to understand the current game state from visual input, determine appropriate actions, and respond with structured outputs that control the game.

For each turn, you will receive:
1. A screenshot of the current game state
2. Contextual information about the game progress, recent events, and objectives

Based on this information, you must analyze the situation, determine the best course of action, and provide a structured JSON response.

## Response Format
Your response MUST follow this exact JSON format with no additional markers, tags, or block delimiters:

{
  "analysis": "Brief analysis of the current game situation, visible UI elements, and important context (1-3 sentences)",
  "current_objective": "The immediate goal based on the game state (single sentence)",
  "reasoning": "Step-by-step logic explaining your chosen action sequence (2-4 sentences)",
  "progress_assessment": "Evaluation of whether previous action(s) achieved their intended goal and why/why not (1-2 sentences)",
  "actions": [
    {
      "type": "press",
      "keys": ["up"|"down"|"left"|"right"|"a"|"b"|"start"|"select"|"pause"]
    },
    {
      "type": "wait",
      "time": milliseconds_to_wait
    }
  ]
}

IMPORTANT: Do not include any conversation markers like <<ASSISTANT_CONVERSATION_START>> or <<ASSISTANT_CONVERSATION_END>> around your response. Provide only the clean JSON object.

## Action Types
- Button presses: {"type": "press", "keys": ["button_name"]} - Valid buttons are: up, down, left, right, a, b, start, select, pause
- Wait for processing: {"type": "wait", "time": milliseconds}

## Important Rules
1. Never use "wait" commands while the game is paused. The game state will not change while paused, so waiting is ineffective.
2. If you detect the game is paused, your next action should be to unpause by using {"type": "press", "keys": ["pause"]} before attempting other actions.
3. Maintain awareness of whether the game is in a paused state based on visual cues in the screenshot.

## Game Play Guidelines
1. **Navigation**: Use directional buttons to move the character or navigate menus
2. **Interaction**: Use 'a' to confirm selections and interact with objects/NPCs, 'b' to cancel or exit menus
3. **Menu Access**: Use 'start' to access the game menu
4. **Battle Strategy**: Analyze Pokémon types, moves, and stats to make optimal battle decisions
5. **Progressive Play**: Work toward completing the current objective while being mindful of longer-term goals like leveling Pokémon, collecting badges, and advancing the story
6. **Resource Management**: Monitor and manage HP, PP, items, and Pokéballs effectively
7. **Memory**: Maintain awareness of the game history and your previous actions to avoid repetitive behaviors

Always provide thoughtful analysis and clear reasoning for your decisions. If you're uncertain about the best course of action, prioritize safe moves that gather more information.
"""  # noqa: E501


def extract_action_from_response_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract actions from a response block.

    Args:
        block: The response block containing actions

    Returns:
        list[dict[str, Any]]: List of actions extracted from the block
    """
    if "actions" in block:
        actions = block["actions"]
        if isinstance(actions, list):
            return actions
    return []


def extract_json_from_response(response: str) -> str:
    """Extract JSON from a response string.

    Args:
        response: The response string containing JSON

    Returns:
        str: The extracted JSON string
    """
    # Try to find JSON block with markdown code block markers
    start = response.find("```json")
    end = response.rfind("```")
    if start != -1 and end != -1:
        start += len("```json")
        return response[start:end].strip()

    # Try to find JSON object directly
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1:
        return response[start : end + 1].strip()

    return response.strip()


class ClaudePlaysPokemon(Agent[AsyncAnthropic, CLA]):
    """AI agent that plays Pokémon games using Claude."""

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        adapter: Adapter | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_message_memory: int = DEFAULT_MAX_MESSAGE_MEMORY,
    ) -> None:
        """Initialize the Claude Plays Pokémon agent.

        Args:
            client: Anthropic API client
            adapter: Game adapter
            model: Claude model to use
            max_tokens: Maximum tokens for response
            max_iterations: Maximum number of iterations
            temperature: Response temperature
            max_message_memory: Maximum number of messages to remember

        Raises:
            ValueError: If API key is not provided
        """
        if client is None:
            api_key = settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key is required")
            client = AsyncAnthropic(api_key=api_key)

        if adapter is None:
            adapter = Adapter()

        super().__init__(
            client=client,
            adapter=adapter,
        )

        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_message_memory = max_message_memory

        self.system_prompts: list[BetaMessageParam] = [
            {
                "role": "assistant",
                "content": generate_system_prompt("Pokemon Red"),
            }
        ]

        self.messages: list[BetaMessageParam] = []

    async def fetch_response(self, observation: Observation) -> tuple[list[dict[str, Any]], bool]:
        """Fetch a response from Claude based on the current observation.

        Args:
            observation: The current game observation

        Returns:
            tuple[list[dict[str, Any]], bool, list[LogType] | None]: List of actions, whether the game is done, and a list of strings or dictionaries of logs.

        Raises:
            ValueError: If client is not initialized
        """  # noqa: E501
        if not self.client:
            raise ValueError("Client is not initialized")

        user_content: list[BetaTextBlockParam | BetaImageBlockParam] = []

        if observation.text:
            user_content.append(
                {
                    "type": "text",
                    "text": observation.text,
                }
            )

        if observation.screenshot:
            logger.debug("Processing screenshot data")
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": observation.screenshot,
                    },
                }
            )

        self.messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        logger.debug(
            "Sending messages to Claude", extra={"messages": self.system_prompts + self.messages}
        )

        response = await self.client.beta.messages.create(
            model=self.model,
            messages=self.system_prompts + self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_content = response.content
        self.messages.append(
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": response_content,
                },
            )
        )

        # Maintain message memory limit
        while len(self.messages) > self.max_message_memory:
            self.messages.pop(0)

        action_list: list[dict[str, Any]] = []

        # Parse response content to extract actions
        for block in response_content:
            if block.type == "text":
                text_json = extract_json_from_response(block.text)
                try:
                    text = json.loads(text_json)
                    if not isinstance(text, dict):
                        logger.error("Invalid response format", extra={"text": text})
                        raise ValueError("Response is not a dictionary")

                    action_list.extend(extract_action_from_response_block(text))

                except json.JSONDecodeError as e:
                    logger.error(
                        "Failed to parse response", extra={"error": str(e), "text": text_json}
                    )

            else:
                logger.error("Unexpected block type", extra={"type": type(block)})

        logger.debug("Extracted actions", extra={"actions": action_list})

        return action_list, False
