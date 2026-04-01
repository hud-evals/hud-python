"""Permission layer for HUD tools.

Provides a lightweight permission system that hooks into BaseTool's
``before()`` callbacks to gate tool execution.

Three modes:
- ALLOW (default): All tool calls proceed without checks.
- PROMPT: Calls ``on_prompt`` callback for each tool call. The callback
  decides whether to allow or deny. Useful for CLI (ask user) or
  server (webhook) permission flows.
- DENY: Block all tool calls by default. Only tools matching
  ``allowlist`` patterns are permitted.

Usage::

    from hud.native.permissions import PermissionLayer, PermissionMode
    from hud.tools.coding import BashTool

    bash = BashTool()

    # Default: allow everything
    perms = PermissionLayer()
    perms.apply(bash)


    # Prompt mode with CLI callback
    async def ask_user(tool_name, args):
        return input(f"Allow {tool_name}? [y/N] ").lower() == "y"


    perms = PermissionLayer(mode=PermissionMode.PROMPT, on_prompt=ask_user)
    perms.apply(bash)

    # Deny mode with allowlist
    perms = PermissionLayer(
        mode=PermissionMode.DENY,
        allowlist=["grep", "glob", "read"],
    )
    perms.apply(bash)  # bash is not in allowlist -> blocked
"""

from __future__ import annotations

import fnmatch
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from hud.tools.types import ToolError

LOGGER = logging.getLogger(__name__)


class PermissionMode(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    DENY = "deny"


class PermissionLayer:
    """Pluggable permission layer for HUD tools.

    Attributes:
        mode: Permission mode (ALLOW, PROMPT, or DENY).
        allowlist: Tool name patterns that bypass denial (fnmatch).
        denylist: Tool name patterns that are always blocked (fnmatch).
        on_prompt: Async callback ``(tool_name, args) -> bool`` for
            PROMPT mode. Must return True to allow, False to deny.
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.ALLOW,
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
        on_prompt: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None,
    ) -> None:
        self.mode = mode
        self.allowlist = allowlist or []
        self.denylist = denylist or []
        self.on_prompt = on_prompt
        self._session_approvals: set[str] = set()

    def _matches(self, name: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    async def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check whether a tool call is permitted.

        Returns True if allowed, False if denied.
        """
        if self._matches(tool_name, self.denylist):
            return False

        if self.mode == PermissionMode.ALLOW:
            return True

        if self._matches(tool_name, self.allowlist):
            return True

        if self.mode == PermissionMode.DENY:
            return False

        if self.mode == PermissionMode.PROMPT:
            if tool_name in self._session_approvals:
                return True

            if self.on_prompt is None:
                LOGGER.warning(
                    "PROMPT mode but no on_prompt callback set, denying %s",
                    tool_name,
                )
                return False

            approved = await self.on_prompt(tool_name, args)
            if approved:
                self._session_approvals.add(tool_name)
            return approved

        return True

    def apply(self, *tools: Any) -> None:
        """Apply this permission layer to one or more BaseTool instances.

        Registers a ``before()`` callback on each tool that calls
        ``check()`` and raises ``ToolError`` on denial.
        """
        from hud.tools.base import BaseTool

        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise TypeError(f"Expected BaseTool, got {type(tool).__name__}")
            self._register(tool)

    def _register(self, tool: Any) -> None:
        layer = self

        @tool.before
        async def _permission_check(**kwargs: Any) -> dict[str, Any] | None:
            allowed = await layer.check(tool.name, kwargs)
            if not allowed:
                raise ToolError(f"Permission denied: {tool.name}")
            return None

    def reset_session(self) -> None:
        """Clear per-session approval cache."""
        self._session_approvals.clear()


def cli_prompt_callback(tool_name: str, args: dict[str, Any]) -> Awaitable[bool]:
    """Default CLI prompt callback using HUDConsole.

    Asks the user interactively whether to allow a tool call.
    Returns an awaitable bool.
    """

    from hud.utils.hud_console import hud_console

    async def _ask() -> bool:
        import json

        args_preview = json.dumps(args, separators=(",", ":"))
        if len(args_preview) > 80:
            args_preview = args_preview[:77] + "..."
        return hud_console.confirm(f"Allow {tool_name}({args_preview})?", default=True)

    return _ask()
