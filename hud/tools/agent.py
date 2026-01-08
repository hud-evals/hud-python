"""AgentTool - run a Task with an agent as a tool."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, get_args, get_origin

from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent
    from hud.eval.task import Task

__all__ = ["AgentTool"]


def _is_eval_only(param: inspect.Parameter) -> bool:
    """Check if param is eval-only: has None default AND None in type union."""
    if param.default is not None:
        return False
    if param.annotation is inspect.Parameter.empty:
        return False
    origin = get_origin(param.annotation)
    if origin is not None:
        return type(None) in get_args(param.annotation)
    return False


class AgentTool(BaseTool):
    """Tool that runs a Task template with an agent.

    Parameters with `| None = None` are eval-only and hidden from the tool schema.

    Example:
        ```python
        @env.scenario()
        async def investigate(
            issue_id: str,  # Required - orchestrator sees
            expected_cause: str | None = None,  # Eval only - hidden
        ):
            yield {"task": f"Investigate {issue_id}"}


        seer = AgentTool(env("investigate"), model="ft:seer-v2")
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
        if model is None and agent is None:
            raise ValueError("Must provide either 'model' or 'agent'")
        if model is not None and agent is not None:
            raise ValueError("Cannot provide both 'model' and 'agent'")

        self._task = task
        self._model = model
        self._agent_cls = agent
        self._agent_params = agent_params or {}
        self._trace = trace

        # Get visible params from scenario function
        self._visible_params: set[str] = set()
        self._param_schema: dict[str, Any] | None = None

        if task.env and task.scenario:
            scenario_fn = task.env._scenarios.get(task.scenario)
            if scenario_fn:
                sig = inspect.signature(scenario_fn)
                visible = {name: p for name, p in sig.parameters.items() if not _is_eval_only(p)}
                self._visible_params = set(visible.keys())
                self._param_schema = self._build_schema(visible)

        tool_name = name or task.scenario or "agent_tool"
        tool_desc = description or f"Run scenario: {task.scenario}"

        super().__init__(name=tool_name, description=tool_desc)

    def _build_schema(self, params: dict[str, inspect.Parameter]) -> dict[str, Any]:
        """Build JSON schema using Pydantic TypeAdapter."""
        from pydantic import TypeAdapter

        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in params.items():
            if param.annotation is not inspect.Parameter.empty:
                try:
                    adapter = TypeAdapter(param.annotation)
                    properties[name] = adapter.json_schema()
                except Exception:
                    properties[name] = {"type": "string"}
            else:
                properties[name] = {"type": "string"}

            if param.default is inspect.Parameter.empty:
                required.append(name)
            elif param.default is not None:
                properties[name]["default"] = param.default

        return {"type": "object", "properties": properties, "required": required}

    @property
    def mcp(self) -> Any:
        """Get as FastMCP FunctionTool with filtered schema."""
        if not hasattr(self, "_mcp_tool"):
            from fastmcp.tools import FunctionTool

            self._mcp_tool = FunctionTool.from_function(
                self, name=self.name, description=self.description
            )
            if self._param_schema:
                self._mcp_tool.parameters = self._param_schema
        return self._mcp_tool

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the task with a fresh agent."""
        from hud.eval.manager import run_eval

        # Filter to visible params only
        filtered = {k: v for k, v in kwargs.items() if k in self._visible_params}

        # Merge with template args
        base_args = self._task.args or {}
        task = self._task.model_copy(update={"args": {**base_args, **filtered}})

        async with run_eval(task, trace=self._trace) as ctx:
            if self._model:
                from hud.agents import create_agent

                agent = create_agent(self._model, **self._agent_params)
            else:
                agent = self._agent_cls.create(**self._agent_params)  # type: ignore

            result = await agent.run(ctx)
            content = result.content if hasattr(result, "content") and result.content else ""
            return ToolResult(content=[TextContent(type="text", text=content)])
