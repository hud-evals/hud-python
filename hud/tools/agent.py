"""AgentTool - run a Task with an agent as a tool."""

from __future__ import annotations

import contextlib
import inspect
import uuid
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from fastmcp.tools.tool import FunctionTool, ToolResult
from mcp.types import TextContent

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent
    from hud.eval.task import Task

__all__ = ["AgentTool"]


def _is_eval_only(param: inspect.Parameter) -> bool:
    """Check if param is eval-only: has None default AND None in type union.

    Handles both runtime types and string annotations (PEP 563).
    """
    # Must have default of None
    if param.default is not None:
        return False
    if param.annotation is inspect.Parameter.empty:
        return False

    annotation = param.annotation

    # Handle string annotations (from __future__ annotations or quoted)
    if isinstance(annotation, str):
        # Check if it looks like "X | None", "Union[X, None]", or "Optional[X]"
        return (
            "| None" in annotation
            or "None |" in annotation
            or "Optional[" in annotation
            or ("Union[" in annotation and "None" in annotation)
        )

    # Handle runtime type annotations
    origin = get_origin(annotation)

    # Union types (X | None or Union[X, None])
    if origin is Union:
        return type(None) in get_args(annotation)

    # For Python 3.10+ union syntax at runtime (types.UnionType)
    try:
        import types

        if isinstance(annotation, types.UnionType):
            return type(None) in get_args(annotation)
    except (ImportError, AttributeError):
        pass

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
        trace_subagent: bool = False,
    ) -> None:
        if not model and agent is None:
            raise ValueError("Must provide either 'model' or 'agent'")
        if model and agent is not None:
            raise ValueError("Cannot provide both 'model' and 'agent'")

        self._task = task
        self._model = model
        self._agent_cls = agent
        self._agent_params = agent_params or {}
        self._trace = trace
        self._trace_subagent = trace_subagent

        # Get visible params from scenario function
        self._visible_params: set[str] = set()
        self._param_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

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
                    # Handle string annotations
                    annotation = param.annotation
                    if isinstance(annotation, str):
                        # Try to evaluate the annotation
                        try:
                            annotation = eval(annotation)  # noqa: S307
                        except Exception:
                            # Fall back to string type but don't skip required handling
                            annotation = None

                    if annotation is not None:
                        adapter = TypeAdapter(annotation)
                        properties[name] = adapter.json_schema()
                    else:
                        properties[name] = {"type": "string"}
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
    def mcp(self) -> FunctionTool:
        """Get as FastMCP FunctionTool with filtered schema."""
        if not hasattr(self, "_mcp_tool"):
            # Directly instantiate FunctionTool with our callable and schema
            # This bypasses from_function's signature parsing
            self._mcp_tool = FunctionTool(
                name=self.name,
                description=self.description or "",
                parameters=self._param_schema,
                fn=self._execute_with_args,
            )
        return self._mcp_tool

    async def _execute_with_args(self, **kwargs: Any) -> ToolResult:
        """Internal executor that FastMCP calls with parsed arguments."""
        return await self(**kwargs)

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Execute the task with a fresh agent."""
        from hud.eval.context import get_current_trace_id
        from hud.eval.manager import run_eval
        from hud.telemetry.instrument import instrument

        # Filter to visible params only
        filtered = {k: v for k, v in kwargs.items() if k in self._visible_params}

        # Merge with template args
        base_args = self._task.args or {}
        task = self._task.model_copy(update={"args": {**base_args, **filtered}})

        # Use parent trace if available (for hierarchical agents)
        parent_trace_id = get_current_trace_id()

        # If nested (has parent), skip subagent's enter/exit registration
        # Tool calls are still recorded via the shared trace_id's context
        is_nested = parent_trace_id is not None

        # If nested, only create a new sub-trace when explicitly requested.
        use_parent_trace = is_nested and not self._trace_subagent
        should_trace = self._trace_subagent if is_nested else (self._trace or self._trace_subagent)

        trace_id: str | None = None
        if use_parent_trace:
            trace_id = parent_trace_id
        elif self._trace or self._trace_subagent:
            trace_id = str(uuid.uuid4())

        # Wrap execution with instrumentation to mark as subagent
        # Platform uses category="subagent" to detect and render subagent tool calls
        @instrument(category="subagent", name=self.name)
        async def _run_subagent() -> ToolResult:
            nonlocal trace_id
            async with run_eval(
                task,
                trace=should_trace,
                trace_id=trace_id,
                quiet=True,
            ) as ctx:
                trace_id = ctx.trace_id
                if self._model:
                    from hud.agents import create_agent

                    agent = create_agent(self._model, **self._agent_params)
                else:
                    agent = self._agent_cls.create(**self._agent_params)  # type: ignore

                result = await agent.run(ctx)
                content = result.content if hasattr(result, "content") and result.content else ""
                return ToolResult(
                    content=[TextContent(type="text", text=content)],
                    meta={"trace_id": trace_id} if trace_id else None,
                )

        try:
            return await _run_subagent()
        except Exception as e:
            if trace_id:
                with contextlib.suppress(Exception):
                    e.trace_id = trace_id  # type: ignore[attr-defined]
            raise
