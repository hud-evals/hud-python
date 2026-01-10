"""Tests for model resolution and create_agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.agents import create_agent
from hud.agents.resolver import resolve_cls


class TestResolveCls:
    """Tests for resolve_cls function."""

    def test_resolves_known_agent_type(self) -> None:
        """Known AgentType strings resolve to their class."""
        from hud.agents.claude import ClaudeAgent

        cls, gateway_info = resolve_cls("claude")
        assert cls == ClaudeAgent
        assert gateway_info is None

    def test_resolves_openai(self) -> None:
        """Resolves 'openai' to OpenAIAgent."""
        from hud.agents import OpenAIAgent

        cls, _gateway_info = resolve_cls("openai")
        assert cls == OpenAIAgent

    def test_resolves_gemini(self) -> None:
        """Resolves 'gemini' to GeminiAgent."""
        from hud.agents.gemini import GeminiAgent

        cls, _gateway_info = resolve_cls("gemini")
        assert cls == GeminiAgent

    def test_unknown_model_without_gateway_raises(self) -> None:
        """Unknown model with no gateway models raises ValueError."""
        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=[]),
            pytest.raises(ValueError, match="not found"),
        ):
            resolve_cls("unknown-model-xyz")

    def test_resolves_gateway_model(self) -> None:
        """Resolves model found in gateway."""
        from hud.agents import OpenAIAgent

        mock_models = [
            {"id": "gpt-4o", "model": "gpt-4o", "provider": "openai"},
        ]

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=mock_models):
            cls, info = resolve_cls("gpt-4o")
            assert cls == OpenAIAgent
            assert info is not None
            assert info["id"] == "gpt-4o"

    def test_resolves_anthropic_provider_to_claude(self) -> None:
        """Provider 'anthropic' maps to ClaudeAgent."""
        from hud.agents.claude import ClaudeAgent

        mock_models = [
            {"id": "claude-sonnet", "model": "claude-3-sonnet", "provider": "anthropic"},
        ]

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=mock_models):
            cls, _info = resolve_cls("claude-sonnet")
            assert cls == ClaudeAgent

    def test_resolves_unknown_provider_to_openai_compatible(self) -> None:
        """Unknown provider maps to OpenAIChatAgent."""
        from hud.agents.openai_chat import OpenAIChatAgent

        mock_models = [
            {"id": "custom-model", "model": "custom", "provider": "custom-provider"},
        ]

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=mock_models):
            cls, _info = resolve_cls("custom-model")
            assert cls == OpenAIChatAgent


class TestCreateAgent:
    """Tests for create_agent function - gateway-only."""

    def test_creates_with_gateway_client(self) -> None:
        """create_agent always uses gateway routing."""
        from hud.agents import OpenAIAgent

        mock_models = [
            {"id": "gpt-4o", "model": "gpt-4o", "provider": "openai"},
        ]

        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=mock_models),
            patch.object(OpenAIAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client") as mock_build_client,
        ):
            mock_client = MagicMock()
            mock_build_client.return_value = mock_client
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_agent("gpt-4o")

            # Should have set model and model_client
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert "model_client" in call_kwargs
            assert agent == mock_agent

    def test_passes_kwargs_to_create(self) -> None:
        """Extra kwargs are passed to agent.create()."""
        from hud.agents import OpenAIAgent

        mock_models = [
            {"id": "gpt-4o", "model": "gpt-4o", "provider": "openai"},
        ]

        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=mock_models),
            patch.object(OpenAIAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client"),
        ):
            mock_create.return_value = MagicMock()

            create_agent("gpt-4o", temperature=0.5, max_tokens=1000)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 1000

    def test_known_agent_type_also_uses_gateway(self) -> None:
        """Even 'claude' string uses gateway (it's a gateway shortcut)."""
        from hud.agents.claude import ClaudeAgent

        with (
            patch.object(ClaudeAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client") as mock_build_client,
        ):
            mock_client = MagicMock()
            mock_build_client.return_value = mock_client
            mock_create.return_value = MagicMock()

            create_agent("claude")

            # Should still build gateway client
            mock_build_client.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "model_client" in call_kwargs


class TestBuildGatewayClient:
    """Tests for build_gateway_client function."""

    def test_builds_anthropic_client(self) -> None:
        """Builds AsyncAnthropic for anthropic provider."""
        from hud.agents.gateway import build_gateway_client

        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.hud_gateway_url = "https://gateway.hud.ai"

            with patch("anthropic.AsyncAnthropic") as mock_client_cls:
                build_gateway_client("anthropic")
                mock_client_cls.assert_called_once()

    def test_builds_openai_client_for_openai(self) -> None:
        """Builds AsyncOpenAI for openai provider."""
        from hud.agents.gateway import build_gateway_client

        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.hud_gateway_url = "https://gateway.hud.ai"

            with patch("openai.AsyncOpenAI") as mock_client_cls:
                build_gateway_client("openai")
                mock_client_cls.assert_called_once()

    def test_builds_openai_client_for_unknown(self) -> None:
        """Builds AsyncOpenAI for unknown providers (openai-compatible)."""
        from hud.agents.gateway import build_gateway_client

        with patch("hud.settings.settings") as mock_settings:
            mock_settings.api_key = "test-key"
            mock_settings.hud_gateway_url = "https://gateway.hud.ai"

            with patch("openai.AsyncOpenAI") as mock_client_cls:
                build_gateway_client("together")
                mock_client_cls.assert_called_once()
