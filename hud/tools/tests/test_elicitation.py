"""Tests for ElicitTool -- MCP elicitation via BaseTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from hud.tools.elicitation import ElicitTool


@pytest.fixture()
def elicit_tool() -> ElicitTool:
    return ElicitTool()


class TestElicitToolBasics:
    def test_name(self, elicit_tool: ElicitTool) -> None:
        assert elicit_tool.name == "elicit"

    def test_has_description(self, elicit_tool: ElicitTool) -> None:
        assert elicit_tool.description
        assert "input" in elicit_tool.description.lower()

    def test_is_base_tool(self, elicit_tool: ElicitTool) -> None:
        from hud.tools.base import BaseTool

        assert isinstance(elicit_tool, BaseTool)

    def test_mcp_property_returns_function_tool(self, elicit_tool: ElicitTool) -> None:
        mcp = elicit_tool.mcp
        assert mcp.name == "elicit"


class TestElicitToolExecution:
    @pytest.mark.asyncio()
    async def test_accepted_string_response(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        mock_result = MagicMock(spec=AcceptedElicitation)
        mock_result.data = "user's answer"
        ctx.elicit = AsyncMock(return_value=mock_result)

        result = await elicit_tool(message="What is your name?", ctx=ctx)

        ctx.elicit.assert_called_once()
        assert len(result) == 1
        assert result[0].text == "user's answer"

    @pytest.mark.asyncio()
    async def test_accepted_with_value_attr(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        mock_data = MagicMock()
        mock_data.value = "selected option"
        mock_result = MagicMock(spec=AcceptedElicitation)
        mock_result.data = mock_data
        ctx.elicit = AsyncMock(return_value=mock_result)

        result = await elicit_tool(message="Pick one", options=["a", "b"], ctx=ctx)

        assert result[0].text == "selected option"

    @pytest.mark.asyncio()
    async def test_declined(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        ctx.elicit = AsyncMock(return_value=MagicMock(spec=DeclinedElicitation))

        result = await elicit_tool(message="Your name?", ctx=ctx)

        assert "declined" in result[0].text.lower()

    @pytest.mark.asyncio()
    async def test_cancelled(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        ctx.elicit = AsyncMock(return_value=MagicMock(spec=CancelledElicitation))

        result = await elicit_tool(message="Your name?", ctx=ctx)

        assert "cancelled" in result[0].text.lower()

    @pytest.mark.asyncio()
    async def test_elicit_not_supported(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        ctx.elicit = AsyncMock(side_effect=RuntimeError("not supported"))

        result = await elicit_tool(message="Your name?", ctx=ctx)

        assert "not available" in result[0].text.lower()

    @pytest.mark.asyncio()
    async def test_options_passed_as_response_type(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        mock_result = MagicMock(spec=AcceptedElicitation)
        mock_result.data = "option_b"
        ctx.elicit = AsyncMock(return_value=mock_result)

        await elicit_tool(message="Pick", options=["option_a", "option_b"], ctx=ctx)

        call_args = ctx.elicit.call_args
        assert call_args.args[0] == "Pick"
        assert call_args.kwargs["response_type"] == ["option_a", "option_b"]

    @pytest.mark.asyncio()
    async def test_no_options_uses_str_type(self, elicit_tool: ElicitTool) -> None:
        ctx = MagicMock()
        mock_result = MagicMock(spec=AcceptedElicitation)
        mock_result.data = "free text"
        ctx.elicit = AsyncMock(return_value=mock_result)

        await elicit_tool(message="Tell me", ctx=ctx)

        call_args = ctx.elicit.call_args
        assert call_args.args[0] == "Tell me"
        assert call_args.kwargs["response_type"] is str
