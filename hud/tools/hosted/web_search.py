"""Claude's native web search tool for real-time information."""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class WebSearchTool(HostedTool):
    """Claude's native web search tool for real-time information.

    When enabled, Claude can search the web and cite sources in its responses.
    The search happens server-side on Anthropic's infrastructure.

    Pricing: $10 per 1,000 searches + standard token costs.
    Citations are always enabled for web search results.

    See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.CLAUDE: NativeToolSpec(
            api_type="web_search_20250305",
            api_name="web_search",
            hosted=True,
        ),
    }

    def __init__(
        self,
        max_uses: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict[str, str] | None = None,
    ) -> None:
        """Initialize WebSearchTool.

        Args:
            max_uses: Maximum number of searches per request
            allowed_domains: Only include results from these domains
            blocked_domains: Never include results from these domains
            user_location: Localize search results (city, region, country, timezone)
        """
        extra: dict[str, Any] = {}
        if max_uses is not None:
            extra["max_uses"] = max_uses
        if allowed_domains is not None:
            extra["allowed_domains"] = allowed_domains
        if blocked_domains is not None:
            extra["blocked_domains"] = blocked_domains
        if user_location is not None:
            extra["user_location"] = user_location

        instance_specs: NativeToolSpecs | None = None
        if extra:
            instance_specs = {
                AgentType.CLAUDE: NativeToolSpec(
                    api_type="web_search_20250305",
                    api_name="web_search",
                    hosted=True,
                    extra=extra,
                ),
            }

        super().__init__(
            name="web_search",
            title="Web Search",
            description="Search the web for real-time information with citations",
            native_specs=instance_specs,
        )
