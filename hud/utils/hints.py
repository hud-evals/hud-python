from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
logger = logging.getLogger(__name__)


@dataclass
class Hint:
    """User guidance the CLI renders alongside an error."""

    title: str
    message: str
    tips: list[str] | None = None
    docs_url: str | None = None
    command_examples: list[str] | None = None
    code: str | None = None
    context: list[str] | None = None


HUD_API_KEY_MISSING = Hint(
    title="HUD API key required",
    message="Missing or invalid HUD_API_KEY.",
    tips=[
        "Set HUD_API_KEY in your environment or run: hud set HUD_API_KEY=your-key-here",
        "Get a key at https://hud.ai",
        "Check for whitespace or truncation",
    ],
    docs_url=None,
    command_examples=None,
    code="HUD_AUTH_MISSING",
    context=["auth", "hud"],
)

RATE_LIMIT_HIT = Hint(
    title="Rate limit reached",
    message="Too many requests.",
    tips=[
        "Lower --max-concurrent",
        "Add retry delay",
        "Check API quotas",
    ],
    docs_url=None,
    command_examples=None,
    code="RATE_LIMIT",
    context=["network"],
)

PRO_PLAN_REQUIRED = Hint(
    title="Pro plan required",
    message="This feature requires Pro.",
    tips=[
        "Upgrade your plan to continue",
    ],
    docs_url="https://hud.ai/project/billing",
    command_examples=None,
    code="PRO_PLAN_REQUIRED",
    context=["billing", "plan"],
)

CREDITS_EXHAUSTED = Hint(
    title="Credits exhausted",
    message="Your credits are exhausted.",
    tips=[
        "Top up credits or upgrade your plan",
    ],
    docs_url="https://hud.ai/project/billing",
    command_examples=None,
    code="CREDITS_EXHAUSTED",
    context=["billing", "credits"],
)

TOOL_NOT_FOUND = Hint(
    title="Tool not found",
    message="Requested tool doesn't exist.",
    tips=[
        "Check tool name spelling",
        "Run: hud analyze --live <image>",
        "Verify server implements tool",
    ],
    docs_url=None,
    command_examples=None,
    code="TOOL_NOT_FOUND",
    context=["mcp", "tools"],
)

CLIENT_NOT_INITIALIZED = Hint(
    title="Client not initialized",
    message="MCP client must be initialized before use.",
    tips=[
        "Call client.initialize() first",
        "Or use async with client:",
        "Check connection succeeded",
    ],
    docs_url=None,
    command_examples=None,
    code="CLIENT_NOT_INIT",
    context=["mcp", "client"],
)

INVALID_CONFIG = Hint(
    title="Invalid configuration",
    message="Configuration is missing or malformed.",
    tips=[
        "Check JSON syntax",
        "Verify required fields",
        "See examples in docs",
    ],
    docs_url=None,
    command_examples=None,
    code="INVALID_CONFIG",
    context=["config"],
)

ENV_VAR_MISSING = Hint(
    title="Environment variable required",
    message="Required environment variables are missing.",
    tips=[
        "Set required environment variables",
        "Use -e flag: hud deploy . -e VAR_NAME=value",
        "Check Dockerfile for ENV requirements",
    ],
    docs_url=None,
    command_examples=["hud deploy . -e BROWSER_PROVIDER=anchorbrowser"],
    code="ENV_VAR_MISSING",
    context=["env", "config"],
)

MCP_SERVER_ERROR = Hint(
    title="MCP server error",
    message="The MCP server encountered an error.",
    tips=[
        "Check server logs for details",
        "Verify server configuration",
        "Ensure all dependencies are installed",
    ],
    docs_url=None,
    command_examples=["hud serve --verbose"],
    code="MCP_SERVER_ERROR",
    context=["mcp", "server"],
)


def render_hints(hints: Iterable[Hint] | None, *, design: Any | None = None) -> None:
    """Print ``hints`` on ``design`` (default: the CLI console), skipping any that fail."""
    if not hints:
        return

    hud_console = design
    try:
        if hud_console is None:
            from hud.utils.hud_console import hud_console as default_design  # lazy import

            hud_console = default_design
    except Exception:
        return

    for hint in hints:
        try:
            if hint.title and hint.title != hint.message:
                hud_console.warning(f"{hint.title}: {hint.message}")
            else:
                hud_console.warning(hint.message)

            if hint.tips:
                for tip in hint.tips:
                    hud_console.info(f"  • {tip}")

            if hint.command_examples:
                for cmd in hint.command_examples:
                    hud_console.command_example(cmd)

            if hint.docs_url:
                hud_console.link(hint.docs_url)
        except Exception:
            logger.warning("Failed to render hint: %s", hint)
            continue
