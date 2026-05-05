"""Tests for OpenAIChatAgent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as types
import pytest
from openai import AsyncOpenAI

from hud.agents.openai_chat import OpenAIChatAgent
from hud.types import MCPToolCall, MCPToolResult


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_openai_client() -> AsyncOpenAI:
    """Return a minimal AsyncOpenAI stub that skips real HTTP setup."""
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def _make_tool_call(
    tool_id: str = "call_1",
    name: str = "my_tool",
    arguments: str = "{}",
) -> MagicMock:
    tc = MagicMock()
    tc.id = tool_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_chat_response(
    content: str | None = "Hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.reasoning_content = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    choice.prompt_token_ids = None
    choice.token_ids = None

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client() -> AsyncOpenAI:
    return _make_openai_client()


@pytest.fixture
def agent_params() -> dict[str, Any]:
    return {"model": "gpt-4o-mini"}


@pytest.fixture
def agent(mock_openai_client: AsyncOpenAI) -> OpenAIChatAgent:
    return OpenAIChatAgent.create(openai_client=mock_openai_client, model="gpt-4o-mini")


# ===========================================================================
# 1. Initialization
# ===========================================================================


class TestOpenAIChatAgentInit:
    def test_init_with_openai_client(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(openai_client=mock_openai_client)
        assert a.oai is mock_openai_client

    def test_init_with_api_key_and_base_url(self) -> None:
        with patch("hud.agents.openai_chat.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            a = OpenAIChatAgent.create(api_key="sk-test", base_url="http://localhost:8080/v1")
            mock_cls.assert_called_once_with(api_key="sk-test", base_url="http://localhost:8080/v1")
            assert a.config.api_key == "sk-test"
            assert a.config.base_url == "http://localhost:8080/v1"

    def test_init_uses_hud_gateway_when_api_key_set(self) -> None:
        with (
            patch("hud.agents.openai_chat.settings") as mock_settings,
            patch("hud.agents.openai_chat.AsyncOpenAI") as mock_cls,
        ):
            mock_settings.api_key = "hud-key-123"
            mock_settings.hud_gateway_url = "https://inference.hud.ai/v1"
            mock_cls.return_value = MagicMock()
            a = OpenAIChatAgent.create()
            mock_cls.assert_called_once_with(
                api_key="hud-key-123", base_url="https://inference.hud.ai/v1"
            )
            assert a.oai is mock_cls.return_value

    def test_init_raises_without_any_auth(self) -> None:
        with patch("hud.agents.openai_chat.settings") as mock_settings:
            mock_settings.api_key = None
            mock_settings.hud_gateway_url = "https://inference.hud.ai/v1"
            with pytest.raises(ValueError, match="No API key found"):
                OpenAIChatAgent.create()

    def test_init_raises_api_key_conflict_with_gateway(self) -> None:
        with patch("hud.agents.openai_chat.settings") as mock_settings:
            mock_settings.api_key = "hud-key"
            mock_settings.hud_gateway_url = "https://inference.hud.ai/v1"
            with pytest.raises(ValueError, match="not allowed with HUD Gateway"):
                OpenAIChatAgent.create(
                    api_key="sk-other",
                    base_url="https://inference.hud.ai/v1",
                )

    def test_completion_kwargs_stored(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client,
            completion_kwargs={"temperature": 0.7, "max_tokens": 1024},
        )
        assert a.completion_kwargs["temperature"] == 0.7
        assert a.completion_kwargs["max_tokens"] == 1024

    def test_checkpoint_injected_into_extra_body(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client,
            checkpoint="my-checkpoint-v2",
        )
        assert a.completion_kwargs["extra_body"]["checkpoint"] == "my-checkpoint-v2"

    def test_checkpoint_merges_existing_extra_body(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client,
            checkpoint="ckpt",
            completion_kwargs={"extra_body": {"foo": "bar"}},
        )
        assert a.completion_kwargs["extra_body"]["checkpoint"] == "ckpt"
        assert a.completion_kwargs["extra_body"]["foo"] == "bar"

    def test_model_name_default(self, agent: OpenAIChatAgent) -> None:
        assert agent.model_name == "OpenAI Chat"

    def test_agent_type(self) -> None:
        from hud.types import AgentType
        assert OpenAIChatAgent.agent_type() == AgentType.OPENAI_COMPATIBLE


# ===========================================================================
# 2. Schema sanitization
# ===========================================================================


class TestSanitizeSchemaForOpenAI:
    @pytest.fixture(autouse=True)
    def _agent(self, agent: OpenAIChatAgent) -> None:
        self.agent = agent

    def test_simple_object_passthrough(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result["type"] == "object"
        assert result["properties"]["x"] == {"type": "string"}

    def test_anyof_nullable_collapses_to_non_null(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result.get("type") == "string"
        assert "anyOf" not in result

    def test_anyof_all_null_falls_back_to_string(self) -> None:
        schema = {"anyOf": [{"type": "null"}]}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result.get("type") == "string"

    def test_prefix_items_converted_to_array(self) -> None:
        schema = {"prefixItems": [{"type": "integer"}, {"type": "string"}]}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result["type"] == "array"
        assert result["items"] == {"type": "integer"}

    def test_prefix_items_fallback_on_non_dict_item(self) -> None:
        schema = {"prefixItems": ["not_a_dict"]}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_nested_properties_sanitized_recursively(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
        }
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result["properties"]["value"]["type"] == "number"

    def test_nested_items_sanitized_recursively(self) -> None:
        schema = {"type": "array", "items": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert result["items"]["type"] == "boolean"

    def test_supported_keys_preserved(self) -> None:
        schema = {
            "type": "integer",
            "description": "A number",
            "enum": [1, 2, 3],
            "default": 1,
            "minimum": 0,
            "maximum": 100,
            "minItems": 1,
            "maxItems": 10,
        }
        result = self.agent._sanitize_schema_for_openai(schema)
        for key in ("type", "description", "enum", "default", "minimum", "maximum"):
            assert key in result, f"key {key!r} missing"

    def test_unsupported_keys_stripped(self) -> None:
        schema = {"type": "object", "additionalProperties": False, "$schema": "...", "title": "T"}
        result = self.agent._sanitize_schema_for_openai(schema)
        assert "additionalProperties" not in result
        assert "$schema" not in result
        assert "title" not in result

    def test_empty_schema_returns_object(self) -> None:
        result = self.agent._sanitize_schema_for_openai({})
        assert result == {"type": "object"}

    def test_non_dict_input_returned_unchanged(self) -> None:
        assert self.agent._sanitize_schema_for_openai("string") == "string"  # type: ignore[arg-type]


# ===========================================================================
# 3. Tool schema formatting
# ===========================================================================


class TestGetToolSchemas:
    def test_schemas_formatted_as_openai_functions(self, agent: OpenAIChatAgent) -> None:
        raw = [
            {
                "name": "click",
                "description": "Click somewhere",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
        ]
        with patch.object(type(agent).__mro__[1], "get_tool_schemas", return_value=raw):
            result = agent.get_tool_schemas()

        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "click"
        assert tool["function"]["description"] == "Click somewhere"
        assert "parameters" in tool["function"]

    def test_empty_parameters_get_default_object(self, agent: OpenAIChatAgent) -> None:
        raw = [{"name": "noop", "description": "do nothing"}]
        with patch.object(type(agent).__mro__[1], "get_tool_schemas", return_value=raw):
            result = agent.get_tool_schemas()

        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_anyof_in_params_sanitized(self, agent: OpenAIChatAgent) -> None:
        raw = [
            {
                "name": "t",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "val": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                },
            }
        ]
        with patch.object(type(agent).__mro__[1], "get_tool_schemas", return_value=raw):
            result = agent.get_tool_schemas()

        prop = result[0]["function"]["parameters"]["properties"]["val"]
        assert prop.get("type") == "string"
        assert "anyOf" not in prop


# ===========================================================================
# 4. _oai_to_mcp
# ===========================================================================


class TestOaiToMcp:
    def test_basic_conversion(self) -> None:
        tc = _make_tool_call("id_1", "do_thing", '{"key": "val"}')
        result = OpenAIChatAgent._oai_to_mcp(tc)
        assert result.id == "id_1"
        assert result.name == "do_thing"
        assert result.arguments == {"key": "val"}

    def test_empty_arguments_string(self) -> None:
        tc = _make_tool_call(arguments="")
        result = OpenAIChatAgent._oai_to_mcp(tc)
        assert result.arguments == {}

    def test_null_arguments_string(self) -> None:
        tc = _make_tool_call(arguments=None)  # type: ignore[arg-type]
        result = OpenAIChatAgent._oai_to_mcp(tc)
        assert result.arguments == {}

    def test_list_arguments_unwrapped(self) -> None:
        tc = _make_tool_call(arguments='[{"x": 1}]')
        result = OpenAIChatAgent._oai_to_mcp(tc)
        assert result.arguments == {"x": 1}

    def test_non_dict_non_list_args_empty(self) -> None:
        tc = _make_tool_call(arguments='"just_a_string"')
        result = OpenAIChatAgent._oai_to_mcp(tc)
        assert result.arguments == {}


# ===========================================================================
# 5. System messages
# ===========================================================================


class TestGetSystemMessages:
    @pytest.mark.asyncio
    async def test_with_system_prompt(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client, system_prompt="Be helpful."
        )
        msgs = await a.get_system_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."

    @pytest.mark.asyncio
    async def test_without_system_prompt(self, agent: OpenAIChatAgent) -> None:
        msgs = await agent.get_system_messages()
        assert msgs == []


# ===========================================================================
# 6. format_blocks
# ===========================================================================


class TestFormatBlocks:
    @pytest.mark.asyncio
    async def test_text_block(self, agent: OpenAIChatAgent) -> None:
        blocks = [types.TextContent(type="text", text="Hello")]
        result = await agent.format_blocks(blocks)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0] == {"type": "text", "text": "Hello"}

    @pytest.mark.asyncio
    async def test_image_block(self, agent: OpenAIChatAgent) -> None:
        blocks = [types.ImageContent(type="image", data="abc123", mimeType="image/png")]
        result = await agent.format_blocks(blocks)
        item = result[0]["content"][0]
        assert item["type"] == "image_url"
        assert item["image_url"]["url"] == "data:image/png;base64,abc123"

    @pytest.mark.asyncio
    async def test_mixed_blocks(self, agent: OpenAIChatAgent) -> None:
        blocks: list[types.ContentBlock] = [
            types.TextContent(type="text", text="Look:"),
            types.ImageContent(type="image", data="data", mimeType="image/jpeg"),
        ]
        result = await agent.format_blocks(blocks)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_empty_blocks_returns_empty_content(self, agent: OpenAIChatAgent) -> None:
        result = await agent.format_blocks([])
        assert result[0]["role"] == "user"
        assert result[0]["content"] == []


# ===========================================================================
# 7. format_tool_results
# ===========================================================================


class TestFormatToolResults:
    @pytest.mark.asyncio
    async def test_text_result(self, agent: OpenAIChatAgent) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="output")], isError=False
            )
        ]
        msgs = await agent.format_tool_results(calls, results)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "c1"
        assert msgs[0]["content"] == "output"

    @pytest.mark.asyncio
    async def test_multiple_text_parts_joined(self, agent: OpenAIChatAgent) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [
            MCPToolResult(
                content=[
                    types.TextContent(type="text", text="part1"),
                    types.TextContent(type="text", text="part2"),
                ],
                isError=False,
            )
        ]
        msgs = await agent.format_tool_results(calls, results)
        assert msgs[0]["content"] == "part1part2"

    @pytest.mark.asyncio
    async def test_empty_content_defaults_to_success_message(
        self, agent: OpenAIChatAgent
    ) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [MCPToolResult(content=[], isError=False)]
        msgs = await agent.format_tool_results(calls, results)
        assert msgs[0]["content"] == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_image_result_appended_as_user_message(self, agent: OpenAIChatAgent) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [
            MCPToolResult(
                content=[
                    types.ImageContent(type="image", data="imgdata", mimeType="image/png")
                ],
                isError=False,
            )
        ]
        msgs = await agent.format_tool_results(calls, results)
        # Should have both a tool message and a user message for the image
        assert any(m["role"] == "tool" for m in msgs)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert any(
            c.get("type") == "image_url" for c in user_msgs[0]["content"]
        )

    @pytest.mark.asyncio
    async def test_dict_text_content(self, agent: OpenAIChatAgent) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [
            MCPToolResult(
                content=[{"type": "text", "text": "dict output"}],  # type: ignore[list-item]
                isError=False,
            )
        ]
        msgs = await agent.format_tool_results(calls, results)
        assert msgs[0]["content"] == "dict output"

    @pytest.mark.asyncio
    async def test_structured_content_used_when_no_content(
        self, agent: OpenAIChatAgent
    ) -> None:
        calls = [MCPToolCall(id="c1", name="tool", arguments={})]
        results = [
            MCPToolResult(
                content=[],
                structuredContent={"result": {"type": "text", "text": "structured"}},
                isError=False,
            )
        ]
        msgs = await agent.format_tool_results(calls, results)
        assert "structured" in msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_calls_rendered_separately(self, agent: OpenAIChatAgent) -> None:
        calls = [
            MCPToolCall(id="c1", name="tool_a", arguments={}),
            MCPToolCall(id="c2", name="tool_b", arguments={}),
        ]
        results = [
            MCPToolResult(content=[types.TextContent(type="text", text="a")], isError=False),
            MCPToolResult(content=[types.TextContent(type="text", text="b")], isError=False),
        ]
        msgs = await agent.format_tool_results(calls, results)
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert tool_msgs[1]["tool_call_id"] == "c2"


# ===========================================================================
# 8. Error handling in get_response
# ===========================================================================


class TestGetResponseErrorHandling:
    def _agent_with_mock_tools(self, mock_openai_client: AsyncOpenAI) -> OpenAIChatAgent:
        a = OpenAIChatAgent.create(openai_client=mock_openai_client, model="gpt-4o-mini")
        a._available_tools = []
        a._initialized = True
        return a

    @pytest.mark.asyncio
    async def test_api_error_returns_error_result(
        self, mock_openai_client: AsyncOpenAI
    ) -> None:
        a = self._agent_with_mock_tools(mock_openai_client)
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("500 server error")

        with patch.object(a, "get_tool_schemas", return_value=[]):
            result = await a.get_response([])

        assert result.isError is True
        assert result.content is not None and "500 server error" in result.content

    @pytest.mark.asyncio
    async def test_invalid_json_error_gives_truncated_message(
        self, mock_openai_client: AsyncOpenAI
    ) -> None:
        a = self._agent_with_mock_tools(mock_openai_client)
        mock_openai_client.chat.completions.create.side_effect = ValueError("Invalid JSON response")

        with patch.object(a, "get_tool_schemas", return_value=[]):
            result = await a.get_response([])

        assert result.isError is True
        assert result.content is not None and "truncated" in result.content.lower()


# ===========================================================================
# 9. Integration: _invoke_chat_completion / get_response params passthrough
# ===========================================================================


class TestInvokeChatCompletion:
    @pytest.mark.asyncio
    async def test_model_passed_to_openai(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client, model="gpt-4o", completion_kwargs={}
        )
        a._available_tools = []
        a._initialized = True

        mock_openai_client.chat.completions.create.return_value = _make_chat_response()

        with patch.object(a, "get_tool_schemas", return_value=[]):
            await a.get_response([])

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_extra_completion_kwargs_forwarded(
        self, mock_openai_client: AsyncOpenAI
    ) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client,
            model="gpt-4o",
            completion_kwargs={"temperature": 0.2, "max_tokens": 512},
        )
        a._available_tools = []
        a._initialized = True

        mock_openai_client.chat.completions.create.return_value = _make_chat_response()

        with patch.object(a, "get_tool_schemas", return_value=[]):
            await a.get_response([])

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_protected_keys_not_forwarded_from_completion_kwargs(
        self, mock_openai_client: AsyncOpenAI
    ) -> None:
        a = OpenAIChatAgent.create(
            openai_client=mock_openai_client,
            model="gpt-4o",
            completion_kwargs={"model": "ignored", "messages": [], "tools": []},
        )
        a._available_tools = []
        a._initialized = True

        mock_openai_client.chat.completions.create.return_value = _make_chat_response()

        with patch.object(a, "get_tool_schemas", return_value=[]):
            await a.get_response([{"role": "user", "content": "hi"}])

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        # model is always the config model, not the one from completion_kwargs
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_get_response_text_content(self, mock_openai_client: AsyncOpenAI) -> None:
        a = OpenAIChatAgent.create(openai_client=mock_openai_client, model="gpt-4o-mini")
        a._available_tools = []
        a._initialized = True
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(
            content="42 is the answer"
        )

        with patch.object(a, "get_tool_schemas", return_value=[]):
            result = await a.get_response([])

        assert result.content == "42 is the answer"
        assert result.done is False
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_get_response_tool_calls_parsed(self, mock_openai_client: AsyncOpenAI) -> None:
        tc = _make_tool_call("id_tc", "click", '{"x": 100, "y": 200}')
        response = _make_chat_response(content=None, tool_calls=[tc])
        mock_openai_client.chat.completions.create.return_value = response

        a = OpenAIChatAgent.create(openai_client=mock_openai_client, model="gpt-4o-mini")
        a._available_tools = []
        a._initialized = True

        with patch.object(a, "get_tool_schemas", return_value=[]):
            result = await a.get_response([])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "click"
        assert result.tool_calls[0].arguments == {"x": 100, "y": 200}

    @pytest.mark.asyncio
    async def test_finish_reason_length_marks_done(
        self, mock_openai_client: AsyncOpenAI
    ) -> None:
        response = _make_chat_response(finish_reason="length")
        mock_openai_client.chat.completions.create.return_value = response

        a = OpenAIChatAgent.create(openai_client=mock_openai_client, model="gpt-4o-mini")
        a._available_tools = []
        a._initialized = True

        with patch.object(a, "get_tool_schemas", return_value=[]):
            result = await a.get_response([])

        assert result.done is True

    @pytest.mark.asyncio
    async def test_invoke_raises_without_client(self) -> None:
        with patch("hud.agents.openai_chat.settings") as mock_settings:
            mock_settings.api_key = "dummy"
            mock_settings.hud_gateway_url = "http://gateway"
            with patch("hud.agents.openai_chat.AsyncOpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                a = OpenAIChatAgent.create()
                a.oai = None  # type: ignore[assignment]

        with pytest.raises(ValueError, match="openai_client is required"):
            await a._invoke_chat_completion(messages=[], tools=None, extra={})
