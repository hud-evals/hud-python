"""Tests for OllamaAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hud.agents.ollama import OllamaAgent
from hud.types import AgentType


@pytest.fixture()
def _patch_settings():
    """Patch settings so tests don't need real env vars."""
    with (
        patch("hud.agents.openai_chat.settings") as mock_settings,
        patch("hud.agents.ollama.settings") as mock_ollama_settings,
    ):
        mock_settings.api_key = None
        mock_settings.hud_gateway_url = "https://inference.hud.ai"
        mock_ollama_settings.ollama_base_url = "http://localhost:11434/v1"
        yield mock_settings, mock_ollama_settings


@pytest.mark.usefixtures("_patch_settings")
class TestOllamaAgentInit:
    def test_init_defaults(self):
        agent = OllamaAgent.create()
        assert agent.config.model == "llama3.2"
        assert agent.config.api_key == "ollama"
        assert agent.config.base_url == "http://localhost:11434/v1"

    def test_init_custom_base_url(self):
        agent = OllamaAgent.create(base_url="http://myhost:8080/v1")
        assert agent.config.base_url == "http://myhost:8080/v1"

    def test_init_custom_model(self):
        agent = OllamaAgent.create(model="mistral")
        assert agent.config.model == "mistral"

    def test_init_with_openai_client(self):
        mock_client = MagicMock()
        agent = OllamaAgent.create(openai_client=mock_client)
        assert agent.oai is mock_client

    def test_agent_type(self):
        agent = OllamaAgent.create()
        assert agent.agent_type() == AgentType.OLLAMA


@pytest.mark.usefixtures("_patch_settings")
class TestOllamaHealthAndModels:
    @pytest.mark.asyncio()
    async def test_check_health_success(self):
        agent = OllamaAgent.create()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("hud.agents.ollama.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await agent.check_health()
            assert result is True

    @pytest.mark.asyncio()
    async def test_check_health_failure(self):
        agent = OllamaAgent.create()

        with patch("hud.agents.ollama.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await agent.check_health()
            assert result is False

    @pytest.mark.asyncio()
    async def test_list_models_success(self):
        agent = OllamaAgent.create()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("hud.agents.ollama.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            models = await agent.list_models()
            assert models == ["llama3.2:latest", "mistral:latest"]

    @pytest.mark.asyncio()
    async def test_list_models_failure(self):
        agent = OllamaAgent.create()

        with patch("hud.agents.ollama.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            models = await agent.list_models()
            assert models == []


class TestOllamaSettings:
    def test_ollama_base_url_from_settings(self):
        with (
            patch("hud.agents.openai_chat.settings") as mock_chat,
            patch("hud.agents.ollama.settings") as mock_ollama,
        ):
            mock_chat.api_key = None
            mock_chat.hud_gateway_url = "https://inference.hud.ai"
            mock_ollama.ollama_base_url = "http://custom:9999/v1"

            agent = OllamaAgent.create()
            assert agent.config.base_url == "http://custom:9999/v1"


class TestOllamaAgentType:
    def test_agent_type_cls_resolves(self):
        assert AgentType.OLLAMA.cls is OllamaAgent

    def test_agent_type_config_cls(self):
        from hud.agents.types import OllamaConfig

        assert AgentType.OLLAMA.config_cls is OllamaConfig


@pytest.mark.usefixtures("_patch_settings")
class TestOllamaInheritedBehavior:
    @pytest.mark.asyncio()
    async def test_get_response_delegates_to_parent(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Ollama!"
        mock_choice.message.tool_calls = None
        mock_choice.message.reasoning_content = None
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OllamaAgent.create(openai_client=mock_client)
        agent._available_tools = []  # Initialize tools for test
        result = await agent.get_response([{"role": "user", "content": "Hi"}])

        assert result.content == "Hello from Ollama!"
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_format_blocks_inherited(self):
        from mcp import types

        agent = OllamaAgent.create()
        blocks: list[types.ContentBlock] = [types.TextContent(type="text", text="hello")]
        result = await agent.format_blocks(blocks)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["text"] == "hello"

    @pytest.mark.asyncio()
    async def test_format_tool_results_inherited(self):
        from mcp import types

        from hud.types import MCPToolCall, MCPToolResult

        agent = OllamaAgent.create()
        calls = [MCPToolCall(id="tc1", name="test_tool", arguments={})]
        results = [
            MCPToolResult(
                content=[types.TextContent(type="text", text="result")],
            )
        ]
        rendered = await agent.format_tool_results(calls, results)
        assert rendered[0]["role"] == "tool"
        assert rendered[0]["tool_call_id"] == "tc1"
