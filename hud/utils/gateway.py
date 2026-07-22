"""HUD inference gateway: provider clients and the model catalog.

The sibling of :mod:`hud.utils.platform` — that module talks to the platform
API, this one talks to the inference gateway. Agent construction on top of the
gateway lives in :func:`hud.agents.create_agent`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from typing import TypeAlias

    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
    from google.genai import Client as GenaiClient

    GatewayClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | GenaiClient | AsyncOpenAI

_T = TypeVar("_T")


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

        client: GatewayClient = AsyncAnthropic(
            api_key=settings.api_key, base_url=settings.hud_gateway_url
        )
        return mark_gateway_client(client)

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions

        client = genai.Client(
            api_key=settings.api_key,
            http_options=HttpOptions(
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
            ),
        )
        return mark_gateway_client(client)

    # OpenAI-compatible (openai, azure, together, groq, fireworks, etc.)
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.hud_gateway_url)
    return mark_gateway_client(client)


_HUD_GATEWAY_CLIENT_ATTR = "_hud_gateway_client"
# Identity fallback for clients that reject attribute assignment (e.g. bare
# ``object()`` in tests). Real SDK clients take the attribute mark.
_GATEWAY_CLIENT_IDS: set[int] = set()


def mark_gateway_client(client: _T) -> _T:
    """Tag a client built for the HUD gateway so hosted serialization can drop it."""
    try:
        setattr(client, _HUD_GATEWAY_CLIENT_ATTR, True)
    except (AttributeError, TypeError):
        _GATEWAY_CLIENT_IDS.add(id(client))
    return client


def is_gateway_client(client: object | None) -> bool:
    """True when ``client`` is missing or was built by :func:`build_gateway_client`.

    Hosted reconstruction rebuilds a gateway client from the model name, so a
    tagged gateway client is safe to strip from ``hosted_spec``. A true BYOK /
    custom client is not.
    """
    if client is None:
        return True
    if bool(getattr(client, _HUD_GATEWAY_CLIENT_ATTR, False)):
        return True
    return id(client) in _GATEWAY_CLIENT_IDS


@lru_cache(maxsize=1)
def list_gateway_models() -> list[GatewayModelInfo]:
    """Models available through the HUD gateway (the platform model catalog)."""
    payload = PlatformClient.from_settings().get("/models")
    return GatewayModelsResponse.model_validate(payload).items
