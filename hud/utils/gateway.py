"""HUD inference gateway: provider clients, and the model catalog they resolve against.

Agent construction on top of the gateway lives in :func:`hud.agents.create_agent`.
"""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.telemetry.context import get_trace_headers
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
    id: str
    name: str | None = None
    model_name: str | None = None
    sdk_agent_type: str | None = None
    is_trainable: bool = False
    provider: GatewayProviderInfo = Field(default_factory=GatewayProviderInfo)
    created_at: datetime | None = None
    released_at: datetime | None = None
    deprecated_at: datetime | None = None

    @property
    def recency(self) -> datetime:
        """Sort key for "newest first": the model's release, else when it joined the catalog."""
        return self.released_at or self.created_at or datetime.min.replace(tzinfo=UTC)


class GatewayModelsResponse(BaseModel):
    """One page of `GET /models`."""

    items: list[GatewayModelInfo]
    total: int


async def _inject_trace_id(request: httpx.Request) -> None:
    """httpx request hook."""
    request.headers.update(get_trace_headers())


def build_model_client(
    provider: str, *, model: str | None = None, gateway: bool = False
) -> GatewayClient:
    """The provider's own key when set, otherwise the HUD gateway; *gateway* skips the
    provider key. A Bedrock inference-profile ARN as the Anthropic *model* is only
    reachable through Bedrock, so it cannot be combined with *gateway*.
    """
    bedrock = provider == "anthropic" and model is not None and model.startswith("arn:aws:bedrock:")
    if bedrock and gateway:
        raise ValueError(f"{model} is a Bedrock inference profile; it cannot use the HUD gateway")
    if bedrock:
        if not (
            settings.aws_access_key_id and settings.aws_secret_access_key and settings.aws_region
        ):
            raise HudAuthenticationError(
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required "
                "for AWS Bedrock"
            )
        from anthropic import AsyncAnthropicBedrock

        return AsyncAnthropicBedrock(
            aws_access_key=settings.aws_access_key_id,
            aws_secret_key=settings.aws_secret_access_key,
            aws_region=settings.aws_region,
        )
    keys = {
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }
    key = keys[provider]
    if gateway or not key:
        if settings.api_key:
            return build_gateway_client(provider)
        raise HudAuthenticationError(
            f"No API key for {provider}. Set its provider key or HUD_API_KEY."
        )
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=key)
    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions, HttpRetryOptions

        return genai.Client(
            api_key=key,
            http_options=HttpOptions(retry_options=HttpRetryOptions(initial_delay=10)),
        )
    return AsyncOpenAI(api_key=key)


def build_gateway_client(provider: str) -> GatewayClient:
    """A *provider* SDK client pointed at the HUD gateway, with trace headers attached."""
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
                event_hooks={"request": [_inject_trace_id]},
            ),
        )

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions, HttpRetryOptions

        return genai.Client(
            api_key=settings.api_key,
            http_options=HttpOptions(
                retry_options=HttpRetryOptions(initial_delay=10),
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
                client_args={
                    "event_hooks": {
                        "request": [lambda request: request.headers.update(get_trace_headers())]
                    }
                },
                async_client_args={
                    "transport": httpx.AsyncHTTPTransport(),
                    "event_hooks": {"request": [_inject_trace_id]},
                },
            ),
        )

    # OpenAI-compatible (openai, azure, together, groq, fireworks, etc.)
    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.hud_gateway_url,
        http_client=DefaultAsyncHttpxClient(
            event_hooks={"request": [_inject_trace_id]},
        ),
    )


@lru_cache(maxsize=1)
def list_gateway_models(platform: PlatformClient | None = None) -> list[GatewayModelInfo]:
    """Models available through the HUD gateway (the whole platform model catalog),
    as seen by *platform* (the settings' client when omitted)."""
    if platform is None:
        platform = PlatformClient.from_settings()
    models: list[GatewayModelInfo] = []
    while True:
        page = GatewayModelsResponse.model_validate(
            platform.get("/models", params={"limit": 100, "offset": len(models)})
        )
        models.extend(page.items)
        if len(models) >= page.total:
            return models
        if not page.items:
            raise ValueError("Models API returned an empty page before the reported total")


def resolve_gateway_model(model: str, platform: PlatformClient | None = None) -> GatewayModelInfo:
    """The catalog entry for a model id, display name, or slug (case-insensitive;
    a slug's provider prefix is optional)."""
    names = [
        (name, entry)
        for entry in list_gateway_models(platform)
        for name in (
            entry.id,
            entry.name,
            entry.model_name,
            (entry.model_name or "").rsplit("/", 1)[-1],
        )
        if name
    ]
    for name, entry in names:
        if name.lower() == model.lower():
            return entry
    near = difflib.get_close_matches(model, [name for name, _ in names], n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(near)}?" if near else " Run `hud models` to list them."
    raise ValueError(f"Model {model!r} not found in the HUD gateway registry.{hint}")
