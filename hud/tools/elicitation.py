"""Base elicitation tool for interactive agent workflows.

Provides a BaseTool subclass that agents can use to request structured
input from users during task execution, using MCP's native elicitation
protocol (ctx.elicit).

Registered on environments by default and available to any agent running
in a ConversationSession. Works across all deployment surfaces: A2A
(translates to TASK_STATE_INPUT_REQUIRED), CLI (terminal prompt), and
web UI (modal).

Follows the same pattern as Codex's ``request_user_input``, Claude
Code's ``AskUserQuestion``, and Spring AI's ``AskUserQuestionTool``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from fastmcp.server.context import Context

LOGGER = logging.getLogger(__name__)


class ElicitTool(BaseTool):
    """Request structured input from the user during task execution.

    Use this tool when you need additional information, clarification,
    or a decision from the user before proceeding. Supports free-text
    input or selection from a list of options.

    Internally delegates to MCP's ``ctx.elicit()`` protocol, which is
    handled by the client's elicitation handler (A2A adapter, CLI
    prompt, or web UI modal depending on deployment surface).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="elicit",
            title="Elicit User Input",
            description=(
                "Request input from the user. Use when you need clarification, "
                "a decision, or additional information before proceeding. "
                "Provide a clear question in 'message'. Optionally provide "
                "'options' as a list of choices for the user to select from."
            ),
            **kwargs,
        )

    async def __call__(  # type: ignore[override]
        self,
        message: str,
        options: list[str] | None = None,
        *,
        ctx: Context,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute the elicitation request.

        Args:
            message: Human-readable question to present to the user
            options: Optional list of choices for the user to pick from
            ctx: FastMCP Context (injected by DI)
        """
        from mcp.types import TextContent

        try:
            if options:
                result = await ctx.elicit(message, response_type=options)
            else:
                result = await ctx.elicit(message, response_type=str)
        except Exception as e:
            LOGGER.warning("Elicitation not supported by client: %s", e)
            return [TextContent(type="text", text=f"Elicitation not available: {e}")]

        if isinstance(result, AcceptedElicitation):
            data = result.data
            text = str(getattr(data, "value", data))
            return [TextContent(type="text", text=text)]
        if isinstance(result, DeclinedElicitation):
            return [TextContent(type="text", text="[User declined to answer]")]
        if isinstance(result, CancelledElicitation):
            return [TextContent(type="text", text="[User cancelled the operation]")]
        return [TextContent(type="text", text=str(result))]
