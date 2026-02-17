"""Gemini's URL context tool for fetching and including web content."""

from __future__ import annotations

from typing import ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class UrlContextTool(HostedTool):
    """Gemini's URL context tool for fetching and including web content.

    When enabled, allows the model to fetch and include content from URLs
    in its context. The fetching happens server-side.

    See: https://ai.google.dev/gemini-api/docs/url-context
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="url_context", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="url_context", hosted=True),
    }

    def __init__(self) -> None:
        """Initialize UrlContextTool."""
        super().__init__(
            name="url_context",
            title="URL Context",
            description="Fetch and include web content from URLs",
        )
