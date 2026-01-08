"""AgentTool - run a Task with an agent as a tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from hud.eval.task import Task
    from hud.types import AgentType

__all__ = ["AgentTool"]


class AgentTool(BaseTool):
    """Tool that runs a Task template with an agent.

    Takes a Task as a template (typically with empty args) and runs it
    with a fresh agent when called. Call-time kwargs are merged into
    the task's args.

    Works for both local scenarios (defined with @env.scenario) and
    remote scenarios (from connected hubs).

    Example:
        ```python
        # Create task template
        template = env("checkout")  # Task with args={}

        # Wrap in AgentTool
        tool = AgentTool(template, "claude", agent_params={"model": "claude-sonnet-4-5"})

        # Call with args - spawns fresh agent
        result = await tool(user="alice")
        ```
    """

    def __init__(
        self,
        task: Task,
        agent_type: str | AgentType,
        agent_params: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        trace: bool = False,
    ) -> None:
        """Create an AgentTool.

        Args:
            task: Task template (scenario + env, typically with empty args).
            agent_type: Agent type ("claude", "openai", etc.) or AgentType enum.
            agent_params: Parameters passed to agent.create().
            name: Override tool name (defaults to scenario name).
            description: Override tool description.
            trace: Whether to trace the sub-agent's execution.
        """
        self._task = task
        self._agent_type = agent_type
        self._agent_params = agent_params or {}
        self._trace = trace

        tool_name = name or task.scenario or "agent_tool"
        tool_desc = description or f"Run scenario: {task.scenario}"

        super().__init__(name=tool_name, description=tool_desc)

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the task with a fresh agent.

        Args:
            **kwargs: Arguments merged into the template's args.

        Returns:
            ToolResult with the agent's response content.
        """
        from hud.agents import create_agent
        from hud.eval.manager import run_eval

        # Merge call kwargs with template args (None means empty template)
        base_args = self._task.args if self._task.args is not None else {}
        merged_args = {**base_args, **kwargs}
        task = self._task.model_copy(update={"args": merged_args})

        # Run with fresh agent
        async with run_eval(task, trace=self._trace) as ctx:
            agent = create_agent(self._agent_type, **self._agent_params)
            result = await agent.run(ctx)
            content = result.content if hasattr(result, "content") and result.content else ""
            return ToolResult(content=[TextContent(type="text", text=content)])

