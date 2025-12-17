"""Tests for scenario discovery via prompts/resources in analyze_environment()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp import types
from pydantic import AnyUrl

from hud.clients.base import BaseHUDClient

if TYPE_CHECKING:
    from hud.types import MCPToolCall, MCPToolResult


class _MockClient(BaseHUDClient):
    """Minimal BaseHUDClient for testing analyze_environment scenario derivation."""

    def __init__(
        self,
        *,
        prompts: list[types.Prompt],
        resources: list[types.Resource],
    ) -> None:
        super().__init__(mcp_config={"test": {"url": "mock://test"}}, verbose=True)
        self._mock_prompts = prompts
        self._mock_resources = resources
        # Skip initialize() (which fetches telemetry); we just need analyze_environment().
        self._initialized = True

    async def _connect(self, mcp_config: dict[str, dict[str, Any]]) -> None:  # pragma: no cover
        return None

    async def list_tools(self) -> list[types.Tool]:
        return []

    async def _list_resources_impl(self) -> list[types.Resource]:
        return self._mock_resources

    async def _list_prompts_impl(self) -> list[types.Prompt]:
        return self._mock_prompts

    async def _call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:  # pragma: no cover
        raise NotImplementedError

    async def read_resource(self, uri: str) -> types.ReadResourceResult | None:  # pragma: no cover
        return None

    async def _disconnect(self) -> None:  # pragma: no cover
        return None


@pytest.mark.asyncio
async def test_analyze_environment_derives_scenarios_from_scenario_prompt_and_resource() -> None:
    prompts = [
        types.Prompt(
            name="my-env:checkout",
            description="[Setup] Checkout flow",
            arguments=[],
        )
    ]
    resources = [
        types.Resource(
            uri=AnyUrl("my-env:checkout"),
            name="checkout",
            description="[Evaluate] Checkout flow",
        )
    ]

    client = _MockClient(prompts=prompts, resources=resources)
    analysis = await client.analyze_environment()

    assert "scenarios" in analysis
    assert len(analysis["scenarios"]) == 1
    scenario = analysis["scenarios"][0]
    assert scenario["id"] == "my-env:checkout"
    assert scenario["env"] == "my-env"
    assert scenario["name"] == "checkout"
    assert scenario["has_setup_prompt"] is True
    assert scenario["has_evaluate_resource"] is True


@pytest.mark.asyncio
async def test_analyze_environment_scenario_from_setup_only() -> None:
    prompts = [
        types.Prompt(
            name="env-x:only_setup",
            description="[Setup] Setup only scenario",
            arguments=[],
        )
    ]
    resources: list[types.Resource] = []

    client = _MockClient(prompts=prompts, resources=resources)
    analysis = await client.analyze_environment()

    assert len(analysis["scenarios"]) == 1
    scenario = analysis["scenarios"][0]
    assert scenario["id"] == "env-x:only_setup"
    assert scenario["has_setup_prompt"] is True
    assert scenario["has_evaluate_resource"] is False


@pytest.mark.asyncio
async def test_analyze_environment_scenario_from_evaluate_only() -> None:
    prompts: list[types.Prompt] = []
    resources = [
        types.Resource(
            uri=AnyUrl("env-y:only_eval"),
            name="only_eval",
            description="[Evaluate] Evaluate only scenario",
        )
    ]

    client = _MockClient(prompts=prompts, resources=resources)
    analysis = await client.analyze_environment()

    assert len(analysis["scenarios"]) == 1
    scenario = analysis["scenarios"][0]
    assert scenario["id"] == "env-y:only_eval"
    assert scenario["has_setup_prompt"] is False
    assert scenario["has_evaluate_resource"] is True


@pytest.mark.asyncio
async def test_analyze_environment_extracts_scenario_code_from_meta() -> None:
    """Test that scenario code is extracted from the meta field."""
    scenario_code = """@env.scenario()
async def checkout(product_id: str):
    await env.call_tool("navigate", url="/checkout")
    yield "Complete the checkout"
    result = await env.call_tool("check_order")
    yield 1.0 if result else 0.0
"""
    # Use model_validate with _meta alias (Pydantic alias for the meta field)
    prompts = [
        types.Prompt.model_validate(
            {
                "name": "my-env:checkout",
                "description": "[Setup] Checkout flow",
                "arguments": [{"name": "product_id", "required": True}],
                "_meta": {"code": scenario_code},
            }
        )
    ]
    resources = [
        types.Resource.model_validate(
            {
                "uri": "my-env:checkout",
                "name": "checkout",
                "description": "[Evaluate] Checkout flow",
                "_meta": {"code": scenario_code},
            }
        )
    ]

    client = _MockClient(prompts=prompts, resources=resources)
    analysis = await client.analyze_environment()

    assert len(analysis["scenarios"]) == 1
    scenario = analysis["scenarios"][0]
    assert scenario["id"] == "my-env:checkout"
    assert "code" in scenario
    assert scenario["code"] == scenario_code
    assert "async def checkout" in scenario["code"]


@pytest.mark.asyncio
async def test_analyze_environment_extracts_meta_on_prompts_and_resources() -> None:
    """Test that meta field is included in prompts and resources analysis."""
    meta_data = {"code": "test code", "extra": "value"}
    # Use model_validate with _meta alias (Pydantic alias for the meta field)
    prompts = [
        types.Prompt.model_validate(
            {
                "name": "test-prompt",
                "description": "A test prompt",
                "arguments": [],
                "_meta": meta_data,
            }
        )
    ]
    resources = [
        types.Resource.model_validate(
            {
                "uri": "file:///test",
                "name": "test-resource",
                "description": "A test resource",
                "_meta": meta_data,
            }
        )
    ]

    client = _MockClient(prompts=prompts, resources=resources)
    analysis = await client.analyze_environment()

    # Check prompts have meta
    assert len(analysis["prompts"]) == 1
    assert "meta" in analysis["prompts"][0]
    assert analysis["prompts"][0]["meta"] == meta_data

    # Check resources have meta
    assert len(analysis["resources"]) == 1
    assert "meta" in analysis["resources"][0]
    assert analysis["resources"][0]["meta"] == meta_data
