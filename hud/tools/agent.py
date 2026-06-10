"""AgentTool — expose an env task as a tool that runs a sub-agent (v6).

A v5 holdover, re-homed onto the v6 rollout flow: wrap an ``@env.task`` callable
(e.g. ``env("write_section")``) so an orchestrator can call it like a tool. Each
call binds a :class:`~hud.eval.Task`, drives a fresh agent over it, and returns
the agent's answer (``run.trace.content``).

Parameters declared ``name | None = None`` on the underlying scenario are
*eval-only* (hidden from the tool schema), matching the v5 behavior.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import types
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin

from mcp.types import TextContent

from .base import BaseTool

if TYPE_CHECKING:
    from fastmcp.tools import FunctionTool, ToolResult

    from hud.environment.task import _TaskFactory

LOGGER = logging.getLogger("hud.tools.agent")

__all__ = ["AgentTool"]


def _annotation_includes_none(annotation: Any) -> bool:
    if isinstance(annotation, str):
        return (
            "| None" in annotation
            or "None |" in annotation
            or "Optional[" in annotation
            or ("Union[" in annotation and "None" in annotation)
        )
    if get_origin(annotation) is Union or isinstance(annotation, types.UnionType):
        return type(None) in get_args(annotation)
    return False


def _is_eval_only(param: inspect.Parameter) -> bool:
    """Eval-only param: ``None`` default AND ``None`` allowed in its type."""
    if param.default is not None or param.annotation is inspect.Parameter.empty:
        return False
    return _annotation_includes_none(param.annotation)


class AgentTool(BaseTool):
    """Run a task with a sub-agent, exposed as an MCP tool.

    Example::

        @env.task
        async def investigate(issue_id: str, expected_cause: str | None = None):
            yield f"Investigate {issue_id}"
            yield 1.0


        seer = AgentTool(env("investigate"), model="claude-haiku-4-5")
        env.add_tool(seer)
    """

    def __init__(
        self,
        task: _TaskFactory[Any],
        *,
        model: str | None = None,
        agent: Any = None,
        agent_params: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> None:
        if not model and agent is None:
            raise ValueError("AgentTool: provide either 'model' or 'agent'")
        if model and agent is not None:
            raise ValueError("AgentTool: provide only one of 'model' or 'agent'")

        self._task = task
        self._model = model
        self._agent_cls = agent
        self._agent_params = agent_params or {}
        self._max_steps = max_steps

        self._visible_params: set[str] = set()
        self._param_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        if parameters is not None:
            self._param_schema = parameters
        else:
            scenario_fn = self._scenario_fn()
            if scenario_fn is not None:
                visible = {
                    n: p
                    for n, p in inspect.signature(scenario_fn).parameters.items()
                    if not _is_eval_only(p)
                }
                self._visible_params = set(visible)
                self._param_schema = self._build_schema(visible)

        task_id = getattr(task, "id", None) or "agent_tool"
        super().__init__(name=name or task_id, description=description or f"Run task: {task_id}")

    def _scenario_fn(self) -> Any:
        """The original task generator, for deriving the tool's parameter schema.

        Prefer the env's recorded ``@env.scenario`` source; otherwise fall back to
        the ``Task``'s function (``__wrapped__`` unwraps the wire-protocol adapter
        back to the author's generator, so its real parameters are visible).
        """
        env = getattr(self._task, "env", None)
        task_id = getattr(self._task, "id", None)
        fns = getattr(env, "_scenario_fns", None)
        if fns is not None and task_id in fns:
            return fns[task_id]
        func = getattr(self._task, "func", None)
        return getattr(func, "__wrapped__", func)

    def _build_schema(self, params: dict[str, inspect.Parameter]) -> dict[str, Any]:
        from pydantic import TypeAdapter

        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in params.items():
            schema: dict[str, Any] = {"type": "string"}
            if param.annotation is not inspect.Parameter.empty:
                with contextlib.suppress(Exception):
                    schema = TypeAdapter(param.annotation).json_schema()
            properties[name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(name)
            elif param.default is not None:
                properties[name]["default"] = param.default
        return {"type": "object", "properties": properties, "required": required}

    @property
    def mcp(self) -> FunctionTool:
        if not hasattr(self, "_mcp_tool"):
            from fastmcp.tools import FunctionTool

            self._mcp_tool = FunctionTool(
                name=self.name,
                description=self.description or "",
                parameters=self._param_schema,
                fn=self.__call__,
            )
        return self._mcp_tool

    async def __call__(self, **kwargs: Any) -> ToolResult:
        from fastmcp.tools import ToolResult

        from hud.telemetry.instrument import instrument

        visible = self._param_schema.get("properties", {})
        args = {k: v for k, v in kwargs.items() if k in visible} if visible else dict(kwargs)

        @instrument(category="subagent", name=self.name)
        async def _run() -> ToolResult:
            task = cast("Any", self._task)(**args)
            agent = self._make_agent()
            async with task as run:
                await agent(run)
            return ToolResult(content=[TextContent(type="text", text=run.trace.content or "")])

        return await _run()

    def _make_agent(self) -> Any:
        if self._model:
            from hud.agents import create_agent

            return create_agent(self._model, **{"max_steps": self._max_steps, **self._agent_params})
        return self._agent_cls(**self._agent_params)
