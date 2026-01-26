"""Claude's native web fetch tool for retrieving full page content."""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class WebFetchTool(HostedTool):
    """Claude's native web fetch tool for retrieving full page content.

    When enabled, Claude can fetch and analyze full content from URLs and PDFs.
    The fetching happens server-side on Anthropic's infrastructure.

    No additional charges beyond standard token costs.
    Requires beta header: web-fetch-2025-09-10

    Security note: Data exfiltration risk exists when processing untrusted input.

    See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-fetch-tool
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.CLAUDE: NativeToolSpec(
            api_type="web_fetch_20250910",
            api_name="web_fetch",
            hosted=True,
            beta="web-fetch-2025-09-10",
        ),
    }

    def __init__(
        self,
        max_uses: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_content_tokens: int | None = None,
        citations_enabled: bool = False,
    ) -> None:
        """Initialize WebFetchTool.

        Args:
            max_uses: Maximum number of fetches per request
            allowed_domains: Only fetch from these domains
            blocked_domains: Never fetch from these domains
            max_content_tokens: Maximum content length in tokens (truncates if exceeded)
            citations_enabled: Enable citations for fetched content
        """
        extra: dict[str, Any] = {}
        if max_uses is not None:
            extra["max_uses"] = max_uses
        if allowed_domains is not None:
            extra["allowed_domains"] = allowed_domains
        if blocked_domains is not None:
            extra["blocked_domains"] = blocked_domains
        if max_content_tokens is not None:
            extra["max_content_tokens"] = max_content_tokens
        if citations_enabled:
            extra["citations"] = {"enabled": True}

        instance_specs: NativeToolSpecs | None = None
        if extra:
            instance_specs = {
                AgentType.CLAUDE: NativeToolSpec(
                    api_type="web_fetch_20250910",
                    api_name="web_fetch",
                    hosted=True,
                    beta="web-fetch-2025-09-10",
                    extra=extra,
                ),
            }

        super().__init__(
            name="web_fetch",
            title="Web Fetch",
            description="Fetch full content from URLs and PDFs",
            native_specs=instance_specs,
        )
