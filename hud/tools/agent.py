"""AgentTool - run a Task with an agent as a tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent
    from hud.eval.task import Task

__all__ = ["AgentTool"]


class AgentTool(BaseTool):
    """Tool that runs a Task template with an agent.

    Takes a Task as a template (typically with args=None) and runs it
    with a fresh agent when called. Call-time kwargs are merged into
    the task's args.

    Works for both local scenarios (defined with @env.scenario) and
    remote scenarios (from connected hubs).

    Example:
        ```python
        from hud.tools import AgentTool

        # Create task template
        template = env("checkout")  # Task with args=None

        # Option 1: Use built-in agent type
        tool = AgentTool(template, model="claude")

        # Option 2: Use custom agent class
        tool = AgentTool(template, agent=MyCustomAgent)

        # Call with args - spawns fresh agent
        result = await tool(user="alice")
        ```
    """

    def __init__(
        self,
        task: Task,
        *,
        model: str | None = None,
        agent: type[MCPAgent] | None = None,
        agent_params: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        trace: bool = False,
    ) -> None:
        """Create an AgentTool.

        Args:
            task: Task template (scenario + env, typically with args=None).
            model: Agent type string ("claude", "openai", "gemini", etc.).
                Uses the same resolution as hud eval CLI.
            agent: Custom agent class (must have .create() method).
                Use this for custom agent implementations.
            agent_params: Parameters passed to agent.create() (model name, etc.).
            name: Override tool name (defaults to scenario name).
            description: Override tool description.
            trace: Whether to trace the sub-agent's execution.

        Note:
            Must provide either `model` or `agent`, not both.
        """
        if model is None and agent is None:
            raise ValueError("Must provide either 'model' or 'agent'")
        if model is not None and agent is not None:
            raise ValueError("Cannot provide both 'model' and 'agent'")

        self._task = task
        self._model = model
        self._agent_cls = agent
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
        from hud.eval.manager import run_eval

        # Merge call kwargs with template args (None means empty template)
        base_args = self._task.args if self._task.args is not None else {}
        merged_args = {**base_args, **kwargs}
        task = self._task.model_copy(update={"args": merged_args})

        # Run with fresh agent
        async with run_eval(task, trace=self._trace) as ctx:
            # Create agent from model string or custom class
            if self._model is not None:
                from hud.agents import create_agent

                agent = create_agent(self._model, **self._agent_params)
            else:
                # Custom agent class - call .create() directly
                agent = self._agent_cls.create(**self._agent_params)  # type: ignore[union-attr]

            result = await agent.run(ctx)
            content = result.content if hasattr(result, "content") and result.content else ""
            return ToolResult(content=[TextContent(type="text", text=content)])
