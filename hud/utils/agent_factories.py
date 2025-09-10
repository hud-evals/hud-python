"""Factory functions for creating agents compatible with run_dataset."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from hud.agents.grounded_openai import GroundedOpenAIChatAgent
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.tools.grounding import GrounderConfig


def create_openai_agent(**kwargs: Any) -> GenericOpenAIChatAgent:
    """Factory for GenericOpenAIChatAgent with run_dataset compatibility.

    Args:
        api_key: OpenAI API key
        base_url: Optional custom API endpoint
        model_name: Model to use (e.g., "gpt-4o-mini")
        **kwargs: Additional arguments passed to GenericOpenAIChatAgent

    Returns:
        Configured GenericOpenAIChatAgent instance

    Example:
        >>> from hud.datasets import run_dataset
        >>> from hud.utils.agent_factories import create_openai_agent
        >>> results = await run_dataset(
        ...     "My Eval",
        ...     "hud-evals/SheetBench-50",
        ...     create_openai_agent,
        ...     {"api_key": "your-key", "model_name": "gpt-4o-mini"},
        ... )
    """
    api_key = kwargs.pop("api_key", None)
    base_url = kwargs.pop("base_url", None)

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return GenericOpenAIChatAgent(openai_client=openai_client, **kwargs)


def create_grounded_agent(**kwargs: Any) -> GroundedOpenAIChatAgent:
    """Factory for GroundedOpenAIChatAgent with run_dataset compatibility.

    Args:
        api_key: OpenAI API key for planning model
        base_url: Optional custom API endpoint for planning model
        model_name: Planning model to use (e.g., "gpt-4o-mini")
        grounder_api_key: API key for grounding model
        grounder_api_base: API base URL for grounding model (default: OpenRouter)
        grounder_model: Grounding model to use (default: qwen/qwen-2.5-vl-7b-instruct)
        **kwargs: Additional arguments passed to GroundedOpenAIChatAgent

    Returns:
        Configured GroundedOpenAIChatAgent instance

    Example:
        >>> from hud.datasets import run_dataset
        >>> from hud.utils.agent_factories import create_grounded_agent
        >>> results = await run_dataset(
        ...     "Grounded Eval",
        ...     dataset,
        ...     create_grounded_agent,
        ...     {
        ...         "api_key": "openai-key",
        ...         "grounder_api_key": "openrouter-key",
        ...         "model_name": "gpt-4o-mini",
        ...     },
        ... )
    """
    api_key = kwargs.pop("api_key", None)
    base_url = kwargs.pop("base_url", None)
    grounder_api_key = kwargs.pop("grounder_api_key", None)
    grounder_api_base = kwargs.pop("grounder_api_base", "https://openrouter.ai/api/v1")
    grounder_model = kwargs.pop("grounder_model", "qwen/qwen-2.5-vl-7b-instruct")

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    grounder_config = GrounderConfig(
        api_base=grounder_api_base, model=grounder_model, api_key=grounder_api_key
    )

    return GroundedOpenAIChatAgent(
        openai_client=openai_client, grounder_config=grounder_config, **kwargs
    )
