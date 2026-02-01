"""Gateway client utilities for HUD inference gateway."""

from __future__ import annotations

from typing import Any


def build_gateway_client(provider: str, *, byok: bool = False) -> Any:
    """Build a client configured for HUD gateway routing.

    Args:
        provider: Provider name ("anthropic", "openai", "gemini", etc.)
        byok: If True, include user's provider API key as BYOK header

    Returns:
        Configured async client for the provider.

    Raises:
        ValueError: If byok=True but provider API key is not configured.
    """
    from hud.settings import settings

    provider = provider.lower()

    # Build BYOK headers if enabled
    extra_headers: dict[str, str] = {}
    if byok:
        env_var = f"{provider.upper()}_API_KEY"
        attr = f"{provider}_api_key"
        byok_key = getattr(settings, attr, None)
        if not byok_key:
            raise ValueError(
                f"BYOK requested but {env_var} is not set. "
                f"Set it with: hud set {env_var}=your-key-here"
            )
        extra_headers[f"x-{provider}-key"] = byok_key

    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(
            api_key=settings.api_key,
            base_url=settings.hud_gateway_url,
            default_headers=extra_headers if extra_headers else None,
        )

    if provider == "gemini":
        from google import genai
        from google.genai.types import HttpOptions

        headers = {"Authorization": f"Bearer {settings.api_key}"}
        headers.update(extra_headers)

        return genai.Client(
            api_key="PLACEHOLDER",
            http_options=HttpOptions(
                api_version="v1beta",
                base_url=settings.hud_gateway_url,
                headers=headers,
            ),
        )

    # OpenAI-compatible
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.hud_gateway_url,
        default_headers=extra_headers if extra_headers else None,
    )
