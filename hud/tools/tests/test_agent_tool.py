"""Tests for AgentTool's public tool schema behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hud.environment import Environment
from hud.eval.task import Task
from hud.tools.agent import AgentTool


class TestAgentToolInit:
    def test_requires_model_or_agent(self) -> None:
        task = Task(args={})

        with pytest.raises(ValueError, match="Must provide either"):
            AgentTool(task)

    def test_cannot_provide_both_model_and_agent(self) -> None:
        task = Task(args={})
        mock_agent = MagicMock()

        with pytest.raises(ValueError, match="Cannot provide both"):
            AgentTool(task, model="claude", agent=mock_agent)  # type: ignore[arg-type]

    def test_name_defaults_to_scenario(self) -> None:
        task = Task(scenario="investigate", args={})
        tool = AgentTool(task, model="claude")

        assert tool.name == "investigate"

    def test_name_can_be_overridden(self) -> None:
        task = Task(scenario="investigate", args={})
        tool = AgentTool(task, model="claude", name="custom_name")

        assert tool.name == "custom_name"


class TestAgentToolMCP:
    def test_mcp_tool_exposes_required_and_defaulted_scenario_parameters(self) -> None:
        env = Environment("test")

        @env.scenario()
        async def investigate(issue_id: str, verbose: bool = False, limit: int = 10):
            yield {"task": f"Investigate {issue_id} {verbose} {limit}"}

        task = env("investigate")
        tool = AgentTool(task, model="claude")

        schema = tool.mcp.parameters
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"issue_id", "verbose", "limit"}
        assert "issue_id" in schema["required"]
        assert "verbose" not in schema["required"]  # Has default
        assert "limit" not in schema["required"]
        assert schema["properties"]["verbose"]["default"] is False
        assert schema["properties"]["limit"]["default"] == 10

    def test_mcp_tool_hides_eval_only_parameters(self) -> None:
        env = Environment("test")

        @env.scenario()
        async def check(
            item_id: str,
            expected_status: str | None = None,  # Eval only
        ):
            yield {"task": f"Check {item_id}"}

        task = env("check")
        tool = AgentTool(task, model="claude")

        schema = tool.mcp.parameters
        assert "item_id" in schema["properties"]
        assert "expected_status" not in schema["properties"]

    def test_mcp_property_returns_tool(self) -> None:
        from fastmcp.tools import FunctionTool

        env = Environment("test")

        @env.scenario()
        async def greet(name: str):
            yield {"task": f"Greet {name}"}

        task = env("greet")
        tool = AgentTool(task, model="claude")

        mcp_tool = tool.mcp
        assert isinstance(mcp_tool, FunctionTool)
