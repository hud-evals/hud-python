"""Tests for AgentTool - scenario-to-agent composition."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hud.environment import Environment
from hud.eval.task import Task
from hud.tools.agent import AgentTool, _is_eval_only


class TestIsEvalOnly:
    """Tests for _is_eval_only helper function."""

    def test_required_param_not_eval_only(self) -> None:
        """Required params (no default) are not eval-only."""

        def fn(x: str) -> None:
            pass

        sig = inspect.signature(fn)
        param = sig.parameters["x"]
        assert not _is_eval_only(param)

    def test_optional_with_value_not_eval_only(self) -> None:
        """Optional params with non-None default are not eval-only."""

        def fn(x: str = "default") -> None:
            pass

        sig = inspect.signature(fn)
        param = sig.parameters["x"]
        assert not _is_eval_only(param)

    def test_optional_none_without_union_not_eval_only(self) -> None:
        """Optional with None default but no None in type is not eval-only."""

        def fn(x: str = None) -> None:  # type: ignore[assignment]  # noqa: RUF013
            pass

        sig = inspect.signature(fn)
        param = sig.parameters["x"]
        assert not _is_eval_only(param)

    def test_optional_none_with_union_is_eval_only(self) -> None:
        """Params with `X | None = None` pattern are eval-only."""

        def fn(x: str | None = None) -> None:
            pass

        sig = inspect.signature(fn)
        param = sig.parameters["x"]
        assert _is_eval_only(param)

    def test_optional_int_none_is_eval_only(self) -> None:
        """Works with int | None = None too."""

        def fn(x: int | None = None) -> None:
            pass

        sig = inspect.signature(fn)
        param = sig.parameters["x"]
        assert _is_eval_only(param)

    def test_string_annotation_with_none_union(self) -> None:
        """Handles string annotations like 'str | None'."""
        # Simulate string annotation
        param = inspect.Parameter(
            "x",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation="str | None",
        )
        assert _is_eval_only(param)

    def test_string_annotation_without_none(self) -> None:
        """String annotations without None are not eval-only."""
        param = inspect.Parameter(
            "x",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation="str",
        )
        assert not _is_eval_only(param)


class TestAgentToolInit:
    """Tests for AgentTool initialization."""

    def test_requires_model_or_agent(self) -> None:
        """Must provide either model or agent."""
        task = Task(args={})

        with pytest.raises(ValueError, match="Must provide either"):
            AgentTool(task)

    def test_cannot_provide_both_model_and_agent(self) -> None:
        """Cannot provide both model and agent."""
        task = Task(args={})
        mock_agent = MagicMock()

        with pytest.raises(ValueError, match="Cannot provide both"):
            AgentTool(task, model="claude", agent=mock_agent)  # type: ignore[arg-type]

    def test_accepts_model_string(self) -> None:
        """Can create with model string."""
        task = Task(scenario="test", args={})
        tool = AgentTool(task, model="claude")

        assert tool._model == "claude"
        assert tool._agent_cls is None

    def test_accepts_agent_class(self) -> None:
        """Can create with custom agent class."""
        task = Task(scenario="test", args={})
        mock_agent_cls = MagicMock()
        tool = AgentTool(task, agent=mock_agent_cls)  # type: ignore[arg-type]

        assert tool._model is None
        assert tool._agent_cls is mock_agent_cls

    def test_name_defaults_to_scenario(self) -> None:
        """Tool name defaults to scenario name."""
        task = Task(scenario="investigate", args={})
        tool = AgentTool(task, model="claude")

        assert tool.name == "investigate"

    def test_name_can_be_overridden(self) -> None:
        """Tool name can be overridden."""
        task = Task(scenario="investigate", args={})
        tool = AgentTool(task, model="claude", name="custom_name")

        assert tool.name == "custom_name"


class TestAgentToolParamFiltering:
    """Tests for parameter filtering (eval-only params hidden)."""

    def test_filters_eval_only_params(self) -> None:
        """Eval-only params (| None = None) are filtered from visible_params."""
        env = Environment("test")

        # Use Union syntax for consistency across Python versions
        @env.scenario()
        async def investigate(
            issue_id: str,
            include_traces: bool = True,
            expected_cause: str | None = None,  # Eval only
        ):
            yield {"task": f"Investigate {issue_id}"}

        task = env("investigate")
        tool = AgentTool(task, model="claude")

        # visible_params should only have issue_id and include_traces
        assert "issue_id" in tool._visible_params
        assert "include_traces" in tool._visible_params
        assert "expected_cause" not in tool._visible_params

    def test_all_required_params_visible(self) -> None:
        """All required params are visible."""
        env = Environment("test")

        @env.scenario()
        async def search(query: str, limit: int):
            yield {"task": f"Search: {query}"}

        task = env("search")
        tool = AgentTool(task, model="claude")

        assert "query" in tool._visible_params
        assert "limit" in tool._visible_params

    def test_optional_with_default_visible(self) -> None:
        """Optional params with non-None defaults are visible."""
        env = Environment("test")

        @env.scenario()
        async def fetch(url: str, request_timeout: int = 30, retries: int = 3):
            yield {"task": f"Fetch {url}"}

        task = env("fetch")
        tool = AgentTool(task, model="claude")

        assert "url" in tool._visible_params
        assert "request_timeout" in tool._visible_params
        assert "retries" in tool._visible_params


class TestAgentToolSchema:
    """Tests for JSON schema generation."""

    def test_builds_json_schema(self) -> None:
        """Builds proper JSON schema from visible params."""
        env = Environment("test")

        @env.scenario()
        async def investigate(issue_id: str, verbose: bool = False):
            yield {"task": f"Investigate {issue_id}"}

        task = env("investigate")
        tool = AgentTool(task, model="claude")

        schema = tool._param_schema
        assert schema is not None
        assert schema["type"] == "object"
        assert "issue_id" in schema["properties"]
        assert "verbose" in schema["properties"]
        assert "issue_id" in schema["required"]
        assert "verbose" not in schema["required"]  # Has default

    def test_schema_excludes_eval_only(self) -> None:
        """Schema excludes eval-only params."""
        env = Environment("test")

        @env.scenario()
        async def check(
            item_id: str,
            expected_status: str | None = None,  # Eval only
        ):
            yield {"task": f"Check {item_id}"}

        task = env("check")
        tool = AgentTool(task, model="claude")

        schema = tool._param_schema
        assert schema is not None
        assert "item_id" in schema["properties"]
        assert "expected_status" not in schema["properties"]


class TestAgentToolMCP:
    """Tests for MCP tool integration."""

    def test_mcp_property_returns_tool(self) -> None:
        """The mcp property returns an MCP Tool."""
        from mcp.types import Tool

        env = Environment("test")

        @env.scenario()
        async def greet(name: str):
            yield {"task": f"Greet {name}"}

        task = env("greet")
        tool = AgentTool(task, model="claude")

        mcp_tool = tool.mcp
        assert isinstance(mcp_tool, Tool)

    def test_mcp_has_filtered_parameters(self) -> None:
        """MCP tool has filtered parameter schema."""
        env = Environment("test")

        @env.scenario()
        async def analyze(
            data: str,
            expected_result: str | None = None,  # Eval only
        ):
            yield {"task": f"Analyze {data}"}

        task = env("analyze")
        tool = AgentTool(task, model="claude")

        mcp_tool = tool.mcp
        params = mcp_tool.inputSchema

        assert "data" in params["properties"]
        assert "expected_result" not in params["properties"]


class TestAgentToolCall:
    """Tests for AgentTool.__call__."""

    @pytest.mark.asyncio
    async def test_filters_kwargs_to_visible_only(self) -> None:
        """Call filters kwargs to visible params only."""
        # Import modules first so patches work
        import hud.agents  # noqa: F401
        import hud.eval.manager  # noqa: F401

        env = Environment("test")

        @env.scenario()
        async def process(item: str, expected: str | None = None):
            yield {"task": f"Process {item}"}

        task = env("process")
        tool = AgentTool(task, model="claude")

        # Mock the eval context and agent
        with (
            patch("hud.eval.manager.run_eval") as mock_run_eval,
            patch("hud.agents.create_agent") as mock_create_agent,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_run_eval.return_value = mock_ctx

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=MagicMock(content="result"))
            mock_create_agent.return_value = mock_agent

            # Call with both visible and eval-only params
            await tool(item="test", expected="should_be_filtered")

            # Check that task was created with filtered args
            call_args = mock_run_eval.call_args
            task_arg = call_args[0][0]
            assert "item" in task_arg.args
            assert "expected" not in task_arg.args  # Filtered out

    @pytest.mark.asyncio
    async def test_merges_template_args(self) -> None:
        """Call merges kwargs with template args."""
        # Import modules first so patches work
        import hud.agents  # noqa: F401
        import hud.eval.manager  # noqa: F401

        env = Environment("test")

        @env.scenario()
        async def search(query: str, limit: int = 10):
            yield {"task": f"Search {query}"}

        # Create template with some args pre-filled
        task = env("search", limit=5)
        tool = AgentTool(task, model="claude")

        with (
            patch("hud.eval.manager.run_eval") as mock_run_eval,
            patch("hud.agents.create_agent") as mock_create_agent,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_run_eval.return_value = mock_ctx

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(return_value=MagicMock(content="result"))
            mock_create_agent.return_value = mock_agent

            # Call with additional arg
            await tool(query="test query")

            # Check merged args
            call_args = mock_run_eval.call_args
            task_arg = call_args[0][0]
            assert task_arg.args["query"] == "test query"
            assert task_arg.args["limit"] == 5  # From template
