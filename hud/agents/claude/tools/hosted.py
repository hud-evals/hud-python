"""Claude hosted tools configured by the Claude harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anthropic.types.beta import (
    BetaCitationsConfigParam,
    BetaToolSearchToolBm25_20251119Param,
    BetaToolUnionParam,
    BetaWebFetchTool20250910Param,
    BetaWebSearchTool20250305Param,
)

if TYPE_CHECKING:
    BetaUserLocationParam = Any

from hud.agents.tools import HostedTool


@dataclass(frozen=True, kw_only=True)
class ClaudeHostedTool(HostedTool[BetaToolUnionParam]):
    """Claude-hosted tool configured by the Claude harness."""


@dataclass(frozen=True, kw_only=True)
class ClaudeWebSearchTool(ClaudeHostedTool):
    """Claude web search."""

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: BetaUserLocationParam | None = None

    def to_params(self) -> BetaWebSearchTool20250305Param:
        _validate_domain_filters(self.allowed_domains, self.blocked_domains)
        params = BetaWebSearchTool20250305Param(
            type="web_search_20250305",
            name="web_search",
        )
        if self.max_uses is not None:
            params["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            params["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            params["blocked_domains"] = self.blocked_domains
        if self.user_location is not None:
            params["user_location"] = self.user_location
        return params


@dataclass(frozen=True, kw_only=True)
class ClaudeWebFetchTool(ClaudeHostedTool):
    """Claude web fetch."""

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    max_content_tokens: int | None = None
    citations_enabled: bool = False

    def to_params(self) -> BetaWebFetchTool20250910Param:
        _validate_domain_filters(self.allowed_domains, self.blocked_domains)
        params = BetaWebFetchTool20250910Param(
            type="web_fetch_20250910",
            name="web_fetch",
        )
        if self.max_uses is not None:
            params["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            params["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            params["blocked_domains"] = self.blocked_domains
        if self.max_content_tokens is not None:
            params["max_content_tokens"] = self.max_content_tokens
        if self.citations_enabled:
            params["citations"] = BetaCitationsConfigParam(enabled=True)
        return params


@dataclass(frozen=True, kw_only=True)
class ClaudeToolSearchTool(ClaudeHostedTool):
    """Claude tool search for large tool sets."""

    threshold: int = 10

    def to_params(self) -> BetaToolSearchToolBm25_20251119Param:
        return BetaToolSearchToolBm25_20251119Param(
            type="tool_search_tool_bm25_20251119",
            name="tool_search_tool_bm25",
        )


def _validate_domain_filters(
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> None:
    if allowed_domains and blocked_domains:
        raise ValueError("Use either allowed_domains or blocked_domains, not both.")
