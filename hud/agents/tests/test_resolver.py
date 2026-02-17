"""Tests for model resolution and create_agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.agents import create_agent
from hud.agents.resolver import resolve_cls


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the models cache before each test."""
    import hud.agents.resolver as resolver_module

    resolver_module._models_cache = None


# Mock API response data matching the platform backend format
MOCK_MODELS = [
    {
        "id": "uuid-1",
        "name": "Claude Sonnet 4.5",
        "model_name": "claude-sonnet-4-5",
        "sdk_agent_type": None,
        "provider": {"name": "Anthropic", "default_sdk_agent_type": "claude"},
    },
    {
        "id": "uuid-2",
        "name": "GPT 5.1",
        "model_name": "gpt-5.1",
        "sdk_agent_type": None,
        "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
    },
    {
        "id": "uuid-3",
        "name": "Operator",
        "model_name": "computer-use-preview",
        "sdk_agent_type": "operator",
        "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
    },
    {
        "id": "uuid-4",
        "name": "Gemini 3 Pro",
        "model_name": "gemini-3-pro-preview",
        "sdk_agent_type": None,
        "provider": {"name": "Gemini", "default_sdk_agent_type": "gemini"},
    },
    {
        "id": "uuid-5",
        "name": "Gemini 2.5 Computer Use Preview",
        "model_name": "gemini-2.5-computer-use-preview",
        "sdk_agent_type": "gemini_cua",
        "provider": {"name": "Gemini", "default_sdk_agent_type": "gemini"},
    },
    {
        "id": "uuid-6",
        "name": "Grok 4.1 Fast",
        "model_name": "grok-4-1-fast",
        "sdk_agent_type": None,
        "provider": {"name": "xAI", "default_sdk_agent_type": "openai_compatible"},
    },
]


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

    def test_unknown_model_raises(self) -> None:
        """Unknown model raises ValueError."""
        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS),
            pytest.raises(ValueError, match="not found"),
        ):
            resolve_cls("unknown-model-xyz-123")

    def test_resolves_claude_model(self) -> None:
        """Resolves Claude model to ClaudeAgent via sdk_agent_type."""
        from hud.agents.claude import ClaudeAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("claude-sonnet-4-5")
            assert cls == ClaudeAgent
            assert info is not None
            assert info["model_name"] == "claude-sonnet-4-5"

    def test_resolves_openai_model(self) -> None:
        """Resolves OpenAI model to OpenAIAgent via sdk_agent_type."""
        from hud.agents import OpenAIAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("gpt-5.1")
            assert cls == OpenAIAgent
            assert info is not None

    def test_resolves_operator_model(self) -> None:
        """Resolves OpenAI CUA model to OperatorAgent via sdk_agent_type override."""
        from hud.agents import OperatorAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("computer-use-preview")
            assert cls == OperatorAgent
            assert info is not None
            assert info["sdk_agent_type"] == "operator"

    def test_resolves_gemini_model(self) -> None:
        """Resolves Gemini model to GeminiAgent via provider default."""
        from hud.agents.gemini import GeminiAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("gemini-3-pro-preview")
            assert cls == GeminiAgent
            assert info is not None

    def test_resolves_gemini_cua_model(self) -> None:
        """Resolves Gemini CUA model to GeminiCUAAgent via sdk_agent_type override."""
        from hud.agents.gemini_cua import GeminiCUAAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("gemini-2.5-computer-use-preview")
            assert cls == GeminiCUAAgent
            assert info is not None
            assert info["sdk_agent_type"] == "gemini_cua"

    def test_resolves_openai_compatible_model(self) -> None:
        """Resolves OpenAI-compatible model to OpenAIChatAgent via provider default."""
        from hud.agents.openai_chat import OpenAIChatAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            cls, info = resolve_cls("grok-4-1-fast")
            assert cls == OpenAIChatAgent
            assert info is not None

    def test_sdk_agent_type_overrides_provider_default(self) -> None:
        """Model's sdk_agent_type takes precedence over provider's default."""
        from hud.agents import OperatorAgent

        with patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS):
            # computer-use-preview has sdk_agent_type="operator" but provider default is "openai"
            cls, info = resolve_cls("computer-use-preview")
            assert cls == OperatorAgent
            assert info is not None
            assert info["provider"]["default_sdk_agent_type"] == "openai"
            assert info["sdk_agent_type"] == "operator"


class TestCreateAgent:
    """Tests for create_agent function - gateway-only."""

    def test_creates_with_gateway_client(self) -> None:
        """create_agent always uses gateway routing."""
        from hud.agents import OpenAIAgent

        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS),
            patch.object(OpenAIAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client") as mock_build_client,
        ):
            mock_client = MagicMock()
            mock_build_client.return_value = mock_client
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_agent("gpt-5.1")

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-5.1"
            assert "model_client" in call_kwargs
            assert agent == mock_agent

    def test_passes_kwargs_to_create(self) -> None:
        """Extra kwargs are passed to agent.create()."""
        from hud.agents import OpenAIAgent

        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS),
            patch.object(OpenAIAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client"),
        ):
            mock_create.return_value = MagicMock()

            create_agent("gpt-5.1", temperature=0.5, max_tokens=1000)

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

            mock_build_client.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "model_client" in call_kwargs

    def test_uses_correct_provider_from_gateway_info(self) -> None:
        """Provider name is extracted from gateway info."""
        from hud.agents.claude import ClaudeAgent

        with (
            patch("hud.agents.resolver._fetch_gateway_models", return_value=MOCK_MODELS),
            patch.object(ClaudeAgent, "create") as mock_create,
            patch("hud.agents.gateway.build_gateway_client") as mock_build_client,
        ):
            mock_build_client.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            create_agent("claude-sonnet-4-5")

            mock_build_client.assert_called_once_with("Anthropic")


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
