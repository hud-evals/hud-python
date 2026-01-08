"""Gateway client utilities for HUD inference gateway."""

from __future__ import annotations

from typing import Any


def build_gateway_client(provider: str) -> Any:
    """Build a client configured for HUD gateway routing.

    Args:
        provider: Provider name ("anthropic", "openai", "gemini", etc.)

    Returns:
        Configured async client for the provider.
    """
    from hud.settings import settings

    provider = provider.lower()

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
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=settings.api_key, base_url=settings.hud_gateway_url)

