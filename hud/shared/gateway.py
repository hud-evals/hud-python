"""HUD inference gateway: provider clients and the model catalog.

The sibling of :mod:`hud.shared.platform` — that module talks to the platform
API, this one talks to the inference gateway. Agent construction on top of the
gateway lives in :func:`hud.agents.create_agent`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.shared.platform import PlatformClient

if TYPE_CHECKING:
    from typing import TypeAlias

    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
    from google.genai import Client as GenaiClient

    GatewayClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | GenaiClient | AsyncOpenAI


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
    if not settings.api_key:
        raise ValueError("HUD_API_KEY is required for HUD gateway clients")

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


@lru_cache(maxsize=1)
def list_gateway_models() -> list[GatewayModelInfo]:
    """Models available through the HUD gateway (the platform model catalog)."""
    payload = PlatformClient.from_settings().get("/models/")
    if not isinstance(payload, dict) or "models" not in payload:
        return []
    return GatewayModelsResponse.model_validate(payload).models
