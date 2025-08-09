"""Computer tool wrappers for remote browser environment."""

import logging
from typing import Optional
from hud.tools.computer import HudComputerTool, AnthropicComputerTool, OpenAIComputerTool
from hud.tools.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


def create_computer_tools(executor: BaseExecutor) -> dict:
    """Create computer tools with the given executor.

    Args:
        executor: Browser executor to use for computer control

    Returns:
        Dictionary of tool name to tool instance
    """
    tools = {}

    try:
        # Create HUD computer tool
        hud_tool = HudComputerTool(executor=executor)
        tools["computer"] = hud_tool
        logger.debug("Created HUD computer tool")
    except Exception as e:
        logger.error(f"Failed to create HUD computer tool: {e}")

    try:
        # Create Anthropic computer tool
        anthropic_tool = AnthropicComputerTool(executor=executor)
        tools["anthropic_computer"] = anthropic_tool
        logger.debug("Created Anthropic computer tool")
    except Exception as e:
        logger.error(f"Failed to create Anthropic computer tool: {e}")

    try:
        # Create OpenAI computer tool
        openai_tool = OpenAIComputerTool(executor=executor)
        tools["openai_computer"] = openai_tool
        logger.debug("Created OpenAI computer tool")
    except Exception as e:
        logger.error(f"Failed to create OpenAI computer tool: {e}")

    return tools
