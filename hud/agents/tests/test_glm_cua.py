"""Tests for GLMCUAAgent implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from mcp import types

from hud.agents.glm_cua import (
    GLM_CUA_INSTRUCTIONS,
    PREDEFINED_GLM_FUNCTIONS,
    GLMCUAAgent,
)
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create a stub OpenAI-compatible async client for GLM."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def glm_agent(mock_openai_client: AsyncMock) -> GLMCUAAgent:
    """Create a GLMCUAAgent with a mock client."""
    agent = GLMCUAAgent.create(
        openai_client=mock_openai_client,
        model="glm-4.5v",
    )
    agent._initialized = True
    return agent


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestGLMCUAInit:
    """Test GLMCUAAgent initialization."""

    def test_default_system_prompt(self, mock_openai_client: AsyncMock) -> None:
        agent = GLMCUAAgent.create(openai_client=mock_openai_client)
        assert agent.config.system_prompt == GLM_CUA_INSTRUCTIONS

    def test_custom_system_prompt(self, mock_openai_client: AsyncMock) -> None:
        agent = GLMCUAAgent.create(
            openai_client=mock_openai_client,
            system_prompt="Custom instructions",
        )
        assert agent.config.system_prompt == "Custom instructions"

    def test_default_model(self, mock_openai_client: AsyncMock) -> None:
        agent = GLMCUAAgent.create(openai_client=mock_openai_client)
        assert agent.config.model == "glm-4.5v"

    def test_custom_model(self, mock_openai_client: AsyncMock) -> None:
        agent = GLMCUAAgent.create(
            openai_client=mock_openai_client,
            model="glm-4.6v",
        )
        assert agent.config.model == "glm-4.6v"

    def test_computer_tool_name(self, glm_agent: GLMCUAAgent) -> None:
        assert glm_agent._computer_tool_name == "glm_computer"

    def test_required_tools(self) -> None:
        assert "glm_computer" in GLMCUAAgent.required_tools


# ---------------------------------------------------------------------------
# _process_tool_call - routing predefined functions
# ---------------------------------------------------------------------------


class TestProcessToolCall:
    """Test _process_tool_call routes predefined GLM functions correctly."""

    def test_left_click_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(name="left_click", arguments={"start_box": "[500, 300]"})
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "left_click"
        assert result.arguments["start_box"] == "[500, 300]"

    def test_type_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(name="type", arguments={"content": "hello"})
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "type"
        assert result.arguments["content"] == "hello"

    def test_scroll_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(
            name="scroll",
            arguments={"start_box": "[500, 500]", "direction": "down", "step": 3},
        )
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "scroll"
        assert result.arguments["direction"] == "down"
        assert result.arguments["step"] == 3

    def test_key_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(name="key", arguments={"keys": "ctrl+c"})
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "key"
        assert result.arguments["keys"] == "ctrl+c"

    def test_left_drag_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(
            name="left_drag",
            arguments={"start_box": "[100, 100]", "end_box": "[500, 500]"},
        )
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "left_drag"
        assert result.arguments["start_box"] == "[100, 100]"
        assert result.arguments["end_box"] == "[500, 500]"

    def test_hover_routed(self, glm_agent: GLMCUAAgent) -> None:
        tc = MCPToolCall(name="hover", arguments={"start_box": "[250, 750]"})
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "hover"

    def test_all_predefined_functions_route(self, glm_agent: GLMCUAAgent) -> None:
        """Every predefined GLM function should be routed to glm_computer."""
        for func_name in PREDEFINED_GLM_FUNCTIONS:
            tc = MCPToolCall(name=func_name, arguments={})
            result = glm_agent._process_tool_call(tc)
            assert result.name == "glm_computer", f"{func_name} was not routed"
            assert result.arguments is not None
            assert result.arguments["action"] == func_name

    def test_non_predefined_passthrough(self, glm_agent: GLMCUAAgent) -> None:
        """Non-predefined tools should pass through unchanged."""
        tc = MCPToolCall(name="custom_tool", arguments={"foo": "bar"})
        result = glm_agent._process_tool_call(tc)

        assert result.name == "custom_tool"
        assert result.arguments == {"foo": "bar"}

    def test_glm_computer_direct_call(self, glm_agent: GLMCUAAgent) -> None:
        """Direct glm_computer calls should pass through with XML fix."""
        tc = MCPToolCall(
            name="glm_computer",
            arguments={"action": "left_click", "start_box": "[100, 200]"},
        )
        result = glm_agent._process_tool_call(tc)

        assert result.name == "glm_computer"
        assert result.arguments is not None
        assert result.arguments["action"] == "left_click"


# ---------------------------------------------------------------------------
# _fix_xml_args
# ---------------------------------------------------------------------------


class TestFixXMLArgs:
    """Test XML-style argument fixing."""

    def test_clean_json_unchanged(self, glm_agent: GLMCUAAgent) -> None:
        args = {"action": "left_click", "start_box": "[500, 300]"}
        result = glm_agent._fix_xml_args(args)
        assert result == args

    def test_xml_style_extraction(self, glm_agent: GLMCUAAgent) -> None:
        args = {"action": "left_click\n<arg_key>start_box</arg_key>\n<arg_value>[114, 167]"}
        result = glm_agent._fix_xml_args(args)

        assert result["action"] == "left_click"
        assert result["start_box"] == "[114, 167]"

    def test_non_string_values_unchanged(self, glm_agent: GLMCUAAgent) -> None:
        args = {"step": 5, "action": "scroll"}
        result = glm_agent._fix_xml_args(args)
        assert result == args

    def test_multiple_xml_pairs(self, glm_agent: GLMCUAAgent) -> None:
        args = {
            "action": "left_drag\n"
            "<arg_key>start_box</arg_key>\n<arg_value>[100, 100]\n"
            "<arg_key>end_box</arg_key>\n<arg_value>[500, 500]"
        }
        result = glm_agent._fix_xml_args(args)

        assert result["action"] == "left_drag"
        assert result["start_box"] == "[100, 100]"
        assert result["end_box"] == "[500, 500]"


# ---------------------------------------------------------------------------
# get_response - terminal actions (DONE/FAIL)
# ---------------------------------------------------------------------------


class TestGetResponseTerminalActions:
    """Test that DONE/FAIL terminate the agent loop."""

    @pytest.mark.asyncio
    async def test_done_terminates(self, glm_agent: GLMCUAAgent) -> None:
        # Simulate super().get_response returning a DONE tool call
        mock_response = AgentResponse(
            tool_calls=[MCPToolCall(name="DONE", arguments={})],
            done=False,
        )
        with patch.object(
            type(glm_agent).__bases__[0],
            "get_response",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await glm_agent.get_response([])
            assert response.done is True
            assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_fail_terminates(self, glm_agent: GLMCUAAgent) -> None:
        mock_response = AgentResponse(
            tool_calls=[MCPToolCall(name="FAIL", arguments={})],
            done=False,
        )
        with patch.object(
            type(glm_agent).__bases__[0],
            "get_response",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await glm_agent.get_response([])
            assert response.done is True
            assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_error_response_passthrough(self, glm_agent: GLMCUAAgent) -> None:
        """Error responses should be returned unchanged."""
        mock_response = AgentResponse(isError=True, content="API error")
        with patch.object(
            type(glm_agent).__bases__[0],
            "get_response",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await glm_agent.get_response([])
            assert response.isError is True
            assert response.content == "API error"

    @pytest.mark.asyncio
    async def test_normal_tool_call_processed(self, glm_agent: GLMCUAAgent) -> None:
        """Non-terminal tool calls should be routed normally."""
        mock_response = AgentResponse(
            tool_calls=[MCPToolCall(name="left_click", arguments={"start_box": "[500, 300]"})],
            done=False,
        )
        with patch.object(
            type(glm_agent).__bases__[0],
            "get_response",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await glm_agent.get_response([])
            assert response.done is False
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "glm_computer"
            assert response.tool_calls[0].arguments is not None
            assert response.tool_calls[0].arguments["action"] == "left_click"


# ---------------------------------------------------------------------------
# format_tool_results
# ---------------------------------------------------------------------------


class TestFormatToolResults:
    """Test formatting tool results for the next turn."""

    @pytest.mark.asyncio
    async def test_text_result(self, glm_agent: GLMCUAAgent) -> None:
        tool_calls = [MCPToolCall(name="glm_computer", arguments={"action": "type"})]
        tool_results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="Typed hello")],
                isError=False,
            )
        ]

        messages = await glm_agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # Should contain text content part
        content_parts = messages[0]["content"]
        assert any(p.get("type") == "text" for p in content_parts)

    @pytest.mark.asyncio
    async def test_image_result(self, glm_agent: GLMCUAAgent) -> None:
        tool_calls = [MCPToolCall(name="glm_computer", arguments={"action": "screenshot"})]
        tool_results = [
            MCPToolResult(
                content=[
                    types.ImageContent(type="image", data="fake_base64", mimeType="image/png")
                ],
                isError=False,
            )
        ]

        messages = await glm_agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        content_parts = messages[0]["content"]
        assert any(p.get("type") == "image_url" for p in content_parts)

    @pytest.mark.asyncio
    async def test_mixed_result(self, glm_agent: GLMCUAAgent) -> None:
        tool_calls = [MCPToolCall(name="glm_computer", arguments={"action": "left_click"})]
        tool_results = [
            MCPToolResult(
                content=[
                    types.TextContent(type="text", text="Clicked"),
                    types.ImageContent(type="image", data="screenshot_data", mimeType="image/png"),
                ],
                isError=False,
            )
        ]

        messages = await glm_agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        content_parts = messages[0]["content"]
        assert any(p.get("type") == "text" for p in content_parts)
        assert any(p.get("type") == "image_url" for p in content_parts)

    @pytest.mark.asyncio
    async def test_empty_result_fallback(self, glm_agent: GLMCUAAgent) -> None:
        """Empty content should produce a fallback message."""
        tool_calls = [MCPToolCall(name="glm_computer", arguments={"action": "left_click"})]
        tool_results = [MCPToolResult(content=[], isError=False)]

        messages = await glm_agent.format_tool_results(tool_calls, tool_results)
        assert len(messages) == 1
        content_parts = messages[0]["content"]
        assert any("completed" in p.get("text", "") for p in content_parts)

    @pytest.mark.asyncio
    async def test_glm_name_preserved(self, glm_agent: GLMCUAAgent) -> None:
        """If tool_call has glm_name attr, it should be used in fallback text."""
        tc = MCPToolCall(name="glm_computer", arguments={"action": "hover"})
        tc.glm_name = "hover"  # type: ignore[attr-defined]

        tool_results = [MCPToolResult(content=[], isError=False)]

        messages = await glm_agent.format_tool_results([tc], tool_results)
        content_parts = messages[0]["content"]
        assert any("hover" in p.get("text", "") for p in content_parts)


# ---------------------------------------------------------------------------
# _legacy_native_spec_fallback
# ---------------------------------------------------------------------------


class TestLegacyNativeSpecFallback:
    """Test legacy tool name detection."""

    def test_exact_match(self, glm_agent: GLMCUAAgent) -> None:
        tool = types.Tool(
            name="glm_computer",
            description="GLM computer",
            inputSchema={"type": "object"},
        )
        spec = glm_agent._legacy_native_spec_fallback(tool)
        assert spec is not None
        assert spec.api_name == "glm_computer"

    def test_suffix_match(self, glm_agent: GLMCUAAgent) -> None:
        tool = types.Tool(
            name="env_glm_computer",
            description="GLM computer",
            inputSchema={"type": "object"},
        )
        spec = glm_agent._legacy_native_spec_fallback(tool)
        assert spec is not None

    def test_computer_name_match(self, glm_agent: GLMCUAAgent) -> None:
        tool = types.Tool(
            name="computer",
            description="computer",
            inputSchema={"type": "object"},
        )
        spec = glm_agent._legacy_native_spec_fallback(tool)
        assert spec is not None

    def test_no_match(self, glm_agent: GLMCUAAgent) -> None:
        tool = types.Tool(
            name="unrelated_tool",
            description="Something",
            inputSchema={"type": "object"},
        )
        spec = glm_agent._legacy_native_spec_fallback(tool)
        assert spec is None
