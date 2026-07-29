"""HUD inference gateway: provider clients and the model catalog.

The sibling of :mod:`hud.utils.platform` — that module talks to the platform
API, this one talks to the inference gateway. Agent construction on top of the
gateway lives in :func:`hud.agents.create_agent`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from typing import TypeAlias

    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
    from google.genai import Client as GenaiClient

    GatewayClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | GenaiClient | AsyncOpenAI


class GatewayProviderInfo(BaseModel):
    name: str | None = None


class GatewayModelInfo(BaseModel):
    id: str | None = None
    name: str | None = None
    model_name: str | None = None
    sdk_agent_type: str | None = None
    is_trainable: bool = False
    provider: GatewayProviderInfo = Field(default_factory=GatewayProviderInfo)


class GatewayModelsResponse(BaseModel):
    """`GET /models` — a paginated platform response; only `items` is read."""

    items: list[GatewayModelInfo]


_MODEL_ALIASES: dict[str, str] = {
    "deepseek-v4": "deepseek/deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "glm-5.2": "z-ai/glm-5.2",
    "kimi-2.6": "moonshotai/kimi-k2.6",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "minimax-m3": "MiniMax-M3",
    "minimax-m2.7": "MiniMax-M2.7",
    "minimax-m2.5": "MiniMax-M2.5",
}


def _inject_trace_id(request: httpx.Request) -> None:
    trace_id = get_current_trace_id()
    if trace_id is not None:
        request.headers["Trace-Id"] = trace_id


async def _inject_trace_id_async(request: httpx.Request) -> None:
    _inject_trace_id(request)


def normalize_gateway_model_id(model: str) -> str:
    """Return the canonical HUD gateway model slug for known short aliases."""
    return _MODEL_ALIASES.get(model.lower(), model)


def gateway_model_aliases() -> tuple[str, ...]:
    """Return accepted short aliases for HUD gateway model slugs."""
    return tuple(_MODEL_ALIASES)


def build_gateway_client(provider: str) -> GatewayClient:
    """Build a client configured for HUD gateway routing.

    Args:
        provider: Provider name ("anthropic", "openai", "gemini", etc.)

    Returns:
        Configured async client for the provider.
    """
    # Provider SDK clients bypass hud.utils.requests, so guard here.
    if not settings.api_key:
        raise HudAuthenticationError("HUD_API_KEY is required for HUD gateway clients")

    provider = provider.lower()

    # Anthropic and Gemini SDKs are optional extras; keep those imports on the
    # provider branch so importing gateway utilities does not require both.
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        from anthropic import DefaultAsyncHttpxClient as AnthropicHttpClient

        return AsyncAnthropic(
            api_key=settings.api_key,
            base_url=settings.hud_gateway_url,
            http_client=AnthropicHttpClient(
                event_hooks={"request": [_inject_trace_id_async]},
            ),
        )

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions

        return genai.Client(
            api_key=settings.api_key,
            http_options=HttpOptions(
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
                client_args={"event_hooks": {"request": [_inject_trace_id]}},
                async_client_args={
                    "transport": httpx.AsyncHTTPTransport(),
                    "event_hooks": {"request": [_inject_trace_id_async]},
                },
            ),
        )

    # OpenAI-compatible (openai, azure, together, groq, fireworks, etc.)
    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.hud_gateway_url,
        http_client=DefaultAsyncHttpxClient(
            event_hooks={"request": [_inject_trace_id_async]},
        ),
    )


@lru_cache(maxsize=1)
def list_gateway_models() -> list[GatewayModelInfo]:
    """Models available through the HUD gateway (the platform model catalog)."""
    payload = PlatformClient.from_settings().get("/models")
    return GatewayModelsResponse.model_validate(payload).items
