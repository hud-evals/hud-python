"""Gateway client utilities for HUD inference gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.types import AgentType

if TYPE_CHECKING:
    from typing import TypeAlias

    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
    from google.genai import Client as GenaiClient

    from hud.agents.claude import ClaudeAgent
    from hud.agents.gemini import GeminiAgent
    from hud.agents.openai import OpenAIAgent
    from hud.agents.openai_compatible import OpenAIChatAgent

    GatewayClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | GenaiClient | AsyncOpenAI
    GatewayAgent: TypeAlias = ClaudeAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent


class GatewayProviderInfo(BaseModel):
    name: str | None = None
    default_sdk_agent_type: str | None = None


class GatewayModelInfo(BaseModel):
    id: str | None = None
    name: str | None = None
    model_name: str | None = None
    sdk_agent_type: str | None = None
    provider: GatewayProviderInfo = Field(default_factory=GatewayProviderInfo)


class GatewayModelsResponse(BaseModel):
    models: list[GatewayModelInfo]


def build_gateway_client(provider: str) -> GatewayClient:
    """Build a client configured for HUD gateway routing.

    Args:
        provider: Provider name ("anthropic", "openai", "gemini", etc.)

    Returns:
        Configured async client for the provider.
    """
    provider = provider.lower()

    # Anthropic and Gemini SDKs are optional extras; keep those imports on the
    # provider branch so importing gateway utilities does not require both.
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=settings.api_key, base_url=settings.hud_gateway_url)

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions

        return genai.Client(
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
                headers={"Authorization": f"Bearer {settings.api_key}"},
            ),
        )

    # OpenAI-compatible (openai, azure, together, groq, fireworks, etc.)
    return AsyncOpenAI(api_key=settings.api_key, base_url=settings.hud_gateway_url)


def _fetch_gateway_models() -> list[GatewayModelInfo]:
    """Fetch available models from HUD API."""
    if not settings.api_key:
        return []

    try:
        resp = httpx.get(
            f"{settings.hud_api_url}/models/",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        payload: object = resp.json()
        if not isinstance(payload, dict) or "models" not in payload:
            return []
        return GatewayModelsResponse.model_validate(payload).models
    except Exception:
        return []


def create_agent(model: str, **kwargs: Any) -> GatewayAgent:
    """Create an agent routed through the HUD gateway.

    For direct API access with provider API keys, instantiate the agent classes directly.
    """
    agent_type = next((candidate for candidate in AgentType if candidate.value == model), None)
    if agent_type is not None:
        model_id = model
        provider_name = agent_type.gateway_provider
    else:
        for gateway_model in _fetch_gateway_models():
            if model in (
                gateway_model.id,
                gateway_model.name,
                gateway_model.model_name,
            ):
                agent_str = (
                    gateway_model.sdk_agent_type or gateway_model.provider.default_sdk_agent_type
                )
                if agent_str == "operator":
                    raise ValueError(
                        "Operator agent is no longer supported; use openai with a supported "
                        "OpenAI computer model."
                    )
                if agent_str == "gemini_cua":
                    raise ValueError(
                        "Gemini CUA agent is no longer supported; use gemini with a supported "
                        "Gemini computer-use model."
                    )
                if not isinstance(agent_str, str):
                    raise ValueError(f"Model '{model}' has invalid agent type metadata")

                agent_type = AgentType(agent_str)
                model_id = gateway_model.model_name or model
                provider_name = gateway_model.provider.name or "openai"
                break
        else:
            raise ValueError(f"Model '{model}' not found")

    client = build_gateway_client(provider_name)
    kwargs.setdefault("model", model_id)
    if agent_type == AgentType.OPENAI_COMPATIBLE:
        kwargs.setdefault("openai_client", client)
    else:
        kwargs.setdefault("model_client", client)
        kwargs.setdefault("validate_api_key", False)

    return agent_type.cls.create(**kwargs)
