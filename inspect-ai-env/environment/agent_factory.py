"""
Agent Factory for Inspect AI integration.

Routes model names to appropriate HUD agent implementations.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


def create_agent_for_model(model_name: str, mcp_client: Any, **kwargs: Any) -> Any:
    """
    Create the appropriate HUD agent based on model name.

    Args:
        model_name: The model identifier (e.g., "claude-3-5-sonnet", "gpt-4o")
        mcp_client: MCP client instance (usually NullMCPClient for Inspect AI)
        **kwargs: Additional arguments to pass to the agent constructor

    Returns:
        Instantiated agent (ClaudeAgent, OperatorAgent, or GenericOpenAIChatAgent)

    Raises:
        ValueError: If the model name cannot be routed to an agent
    """
    model_lower = model_name.lower()

    # Route to Claude agent
    if "claude" in model_lower:
        logger.info(f"Routing model '{model_name}' to ClaudeAgent")
        from hud.agents import ClaudeAgent

        return ClaudeAgent(
            mcp_client=mcp_client,
            model=model_name,
            validate_api_key=True,
            **kwargs,
        )

    # Route to Operator agent (OpenAI computer use)
    elif "computer-use" in model_lower or "operator" in model_lower:
        logger.info(f"Routing model '{model_name}' to OperatorAgent")
        from hud.agents import OperatorAgent

        return OperatorAgent(
            mcp_client=mcp_client,
            model=model_name,
            validate_api_key=True,
            **kwargs,
        )

    # Route to generic OpenAI chat agent (gpt models, etc.)
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        logger.info(f"Routing model '{model_name}' to GenericOpenAIChatAgent")
        from hud.agents import GenericOpenAIChatAgent
        from openai import AsyncOpenAI

        # Create OpenAI client
        openai_client = AsyncOpenAI()  # Will use OPENAI_API_KEY from environment

        return GenericOpenAIChatAgent(
            mcp_client=mcp_client,
            openai_client=openai_client,
            model_name=model_name,
            **kwargs,
        )

    # Default to generic OpenAI chat agent
    else:
        logger.warning(
            f"Unknown model '{model_name}', defaulting to GenericOpenAIChatAgent. "
            "This assumes the model is OpenAI-compatible."
        )
        from hud.agents import GenericOpenAIChatAgent
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI()

        return GenericOpenAIChatAgent(
            mcp_client=mcp_client,
            openai_client=openai_client,
            model_name=model_name,
            **kwargs,
        )
