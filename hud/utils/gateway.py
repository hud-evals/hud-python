"""HUD inference gateway: provider clients, and the model catalog they resolve against."""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args

import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, Field

from hud.settings import settings
from hud.telemetry.context import get_trace_headers
from hud.utils.exceptions import HudAuthenticationError, HudRequestError
from hud.utils.platform import Page, PlatformClient

if TYPE_CHECKING:
    # At runtime these load only where used: the SDKs are slow to import, and importing
    # hud loads this module.
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
    from google.genai import Client as GenaiClient

    GatewayClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | GenaiClient | AsyncOpenAI


class GatewayProviderInfo(BaseModel):
    name: str | None = None


class GatewayModelInfo(BaseModel):
    id: str
    name: str | None = None
    model_name: str
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


# The provider APIs the gateway speaks and the SDK has keys and clients for.
ModelProvider: TypeAlias = Literal["anthropic", "gemini", "openai"]


def _require_provider(provider: str) -> None:
    if provider not in get_args(ModelProvider):
        expected = ", ".join(get_args(ModelProvider))
        raise ValueError(f"unknown model provider {provider!r}; expected one of {expected}")


async def _inject_trace_id(request: httpx2.Request) -> None:
    """Attach the current trace headers to an outgoing request."""
    request.headers.update(get_trace_headers())


def provider_api_key(provider: ModelProvider) -> str | None:
    keys = {
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }
    return keys[provider]


def routes_to_gateway(provider: ModelProvider, *, gateway: bool) -> bool:
    """Whether ``provider`` traffic goes through the HUD gateway: when ``gateway`` forces it,
    or when the provider has no key of its own and ``HUD_API_KEY`` is set.
    """
    return gateway or (not provider_api_key(provider) and bool(settings.api_key))


def build_model_client(
    provider: ModelProvider, *, model: str | None = None, gateway: bool = False
) -> GatewayClient:
    """A ``provider`` SDK client using the provider's own key when set, otherwise the HUD
    gateway; ``gateway`` forces the gateway.

    A Bedrock inference-profile ARN as the Anthropic ``model`` needs the AWS credentials
    and cannot use the gateway.
    """
    _require_provider(provider)
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
    if routes_to_gateway(provider, gateway=gateway):
        return build_gateway_client(provider)
    key = provider_api_key(provider)
    if not key:
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
            http_options=HttpOptions(
                api_version="v1beta", retry_options=HttpRetryOptions(initial_delay=10)
            ),
        )
    return AsyncOpenAI(api_key=key)


# The gateway answers 503 + Retry-After when a pod is at capacity; the OpenAI and
# Anthropic SDKs wait the header out, but their default of two retries is too few.
GATEWAY_MAX_RETRIES = 6


def build_gateway_client(provider: ModelProvider) -> GatewayClient:
    """A ``provider`` SDK client pointed at the HUD gateway, with trace headers attached."""
    _require_provider(provider)
    if not settings.api_key:
        raise HudAuthenticationError("HUD_API_KEY is required for HUD gateway clients")

    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        from anthropic import DefaultAsyncHttpxClient as AnthropicHttpClient

        return AsyncAnthropic(
            api_key=settings.api_key,
            base_url=settings.hud_gateway_url,
            max_retries=GATEWAY_MAX_RETRIES,
            http_client=AnthropicHttpClient(
                event_hooks={"request": [_inject_trace_id]},
            ),
        )

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions, HttpRetryOptions

        # google-genai ignores Retry-After, so its backoff is set to the gateway's 3 s.
        return genai.Client(
            api_key=settings.api_key,
            http_options=HttpOptions(
                retry_options=HttpRetryOptions(initial_delay=3, max_delay=5),
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
                client_args={
                    "event_hooks": {
                        "request": [lambda request: request.headers.update(get_trace_headers())]
                    }
                },
                async_client_args={
                    "transport": httpx2.AsyncHTTPTransport(),
                    "event_hooks": {"request": [_inject_trace_id]},
                },
            ),
        )

    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.hud_gateway_url,
        max_retries=GATEWAY_MAX_RETRIES,
        http_client=DefaultAsyncHttpxClient(
            event_hooks={"request": [_inject_trace_id]},
        ),
    )


def list_gateway_models(platform: PlatformClient | None = None) -> list[GatewayModelInfo]:
    """Models available through the HUD gateway (the whole platform model catalog),
    as seen by ``platform`` (the settings' client when omitted)."""
    return _model_catalog(platform or PlatformClient.from_settings())


@cache
def _model_catalog(platform: PlatformClient) -> list[GatewayModelInfo]:
    models: list[GatewayModelInfo] = []
    while True:
        page = platform.get(
            "/models", Page[GatewayModelInfo], params={"limit": 100, "offset": len(models)}
        )
        models.extend(page.items)
        if len(models) >= page.total:
            return models
        if not page.items:
            raise HudRequestError("Models API returned an empty page before the reported total")


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
            entry.model_name.rsplit("/", 1)[-1],
        )
        if name
    ]
    for name, entry in names:
        if name.lower() == model.lower():
            return entry
    near = difflib.get_close_matches(model, [name for name, _ in names], n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(near)}?" if near else " Run `hud models` to list them."
    raise ValueError(f"Model {model!r} not found in the HUD gateway registry.{hint}")
