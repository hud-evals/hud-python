from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from hud.agents.types import ContentBlock, EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastmcp import FastMCP
    from fastmcp.tools import FunctionTool, ToolResult

# Basic result types for tools
BaseResult = list[ContentBlock] | EvaluationResult

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Base helper class for all MCP tools to constrain their output.

    USAGE:
    All tools should inherit from this class and implement the __call__ method.
    Tools are registered with FastMCP using add_tool.

    FORMAT:
    Tools that return messages should return a list[ContentBlock].
    Tools that return miscallaneous content should return a pydantic model such as EvaluationResult.
    Both of these types of tools are processed via structuredContent.
    Any other type of tool will not be processed well by the client.

    Provider-native tool definitions belong to agent harnesses. Environment
    tools expose MCP schemas and optional environment metadata only.
    """

    def __init__(
        self,
        env: Any = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            env: Optional, often stateful, context object that the tool operates on. Could be:
                - A game instance (e.g., Chess Board)
                - An executor (e.g., PyAutoGUIExecutor for computer control)
                - A browser/page instance (e.g., Playwright Page)
                - Any stateful resource the tool needs to interact with
            name: Tool name for MCP registration (auto-generated from class name if not provided)
            title: Human-readable display name for the tool (auto-generated from class name)
            description: Tool description (auto-generated from docstring if not provided)
            meta: Metadata to include in MCP tool listing (e.g., resolution info)
        """
        self.env = env
        self.name = name or self.__class__.__name__.lower().replace("tool", "")
        self.title = title or self.__class__.__name__.replace("Tool", "").replace("_", " ").title()
        self.description = description or (self.__doc__.strip() if self.__doc__ else None)
        self.meta = meta or {}
        self._callbacks: dict[
            str,
            list[Callable[..., Awaitable[Any]]],
        ] = {}  # {"event_name": [callback_functions]}

        # Expose attributes FastMCP expects when registering an instance directly
        self.__name__ = self.name  # FastMCP uses fn.__name__ if name param omitted
        if self.description:
            self.__doc__ = self.description

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Often uses the context to perform an action.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            List of ContentBlock (TextContent, ImageContent, etc.) with the tool's output
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def register(self, server: FastMCP, **meta: Any) -> BaseTool:
        """Register this tool on a FastMCP server and return self for chaining."""
        server.add_tool(self.mcp, **meta)
        return self

    @property
    def mcp(self) -> FunctionTool:
        """Get this tool as a FastMCP FunctionTool (cached).

        This allows clean registration:
            server.add_tool(my_tool.mcp)

        The tool's __call__ is wrapped to trigger before and after callbacks,
        enabling pre-execution validation and post-execution processing.
        """
        if not hasattr(self, "_mcp_tool"):
            from functools import wraps

            from fastmcp.tools.function_tool import FunctionTool

            original_call = self.__call__

            @wraps(original_call)
            async def wrapped_call(**kwargs: Any) -> Any:
                kwargs = await self._run_before(kwargs)
                result = await original_call(**kwargs)
                return await self._run_after(kwargs, result)

            self._mcp_tool = FunctionTool.from_function(
                wrapped_call,
                name=self.name,
                title=self.title,
                description=self.description,
                meta=self.meta,
            )
        return self._mcp_tool

    def before(
        self, fn: Callable[..., Awaitable[dict[str, Any] | None]]
    ) -> Callable[..., Awaitable[dict[str, Any] | None]]:
        """Decorator to run a function before tool execution.

        The callback receives tool kwargs and can:
        - Return modified kwargs (dict) to change arguments
        - Return None to proceed with original kwargs
        - Raise an exception to block execution

        Example:
            ```python
            bash = BashTool()


            @bash.before
            async def validate(command: str | None = None, **kwargs):
                if command and "rm -rf" in command:
                    raise ToolError("Blocked dangerous command")
                return None  # Proceed with original args
            ```
        """
        self._callbacks.setdefault("before", []).append(fn)
        return fn

    def after(self, fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Decorator to run a function after tool execution.

        The callback receives tool kwargs plus `result=` and can:
        - Return modified result to change what's returned
        - Return None to proceed with original result

        Example:
            ```python
            bash = BashTool()


            @bash.after
            async def log_execution(command: str | None = None, result=None, **kwargs):
                logger.info("Executed: %s", command)
                return None  # Keep original result
            ```
        """
        self._callbacks.setdefault("after", []).append(fn)
        return fn

    async def _run_before(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Run before callbacks. Can modify kwargs or raise to block."""
        for callback in self._callbacks.get("before", []):
            result = await callback(**kwargs)
            if result is not None:
                kwargs = result
        return kwargs

    async def _run_after(self, kwargs: dict[str, Any], result: Any) -> Any:
        """Run after callbacks. Can modify result."""
        for callback in self._callbacks.get("after", []):
            try:
                modified = await callback(result=result, **kwargs)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.warning("after callback failed: %s", e)
        return result
