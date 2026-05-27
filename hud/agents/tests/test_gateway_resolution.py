"""HUD gateway agent resolution tests."""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import hud.agents.gateway as gateway_module
from hud.agents import OpenAIAgent, create_agent
from hud.agents.claude import ClaudeAgent
from hud.agents.gateway import GatewayModelsResponse, build_gateway_client
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.types import AgentType

MODELS = GatewayModelsResponse.model_validate(
    {
        "models": [
            {
                "id": "uuid-openai",
                "name": "GPT 5.4",
                "model_name": "gpt-5.4",
                "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
            },
            {
                "id": "uuid-claude",
                "name": "Claude Sonnet 4.6",
                "model_name": "claude-sonnet-4-6",
                "provider": {"name": "Anthropic", "default_sdk_agent_type": "claude"},
            },
            {
                "id": "uuid-grok",
                "name": "Grok 4.1 Fast",
                "model_name": "grok-4-1-fast",
                "provider": {"name": "xAI", "default_sdk_agent_type": "openai_compatible"},
            },
            {
                "id": "uuid-operator",
                "name": "Operator",
                "model_name": "computer-use-preview",
                "sdk_agent_type": "operator",
                "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
            },
            {
                "id": "uuid-gemini-cua",
                "name": "Gemini Computer Use",
                "model_name": "gemini-2.5-computer-use-preview",
                "sdk_agent_type": "gemini_cua",
                "provider": {"name": "Gemini", "default_sdk_agent_type": "gemini"},
            },
        ]
    }
).models


def test_create_agent_resolves_gateway_model_to_provider_agent() -> None:
    expected = MagicMock()
    client = MagicMock()
    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=MODELS),
        patch("hud.agents.gateway.build_gateway_client", return_value=client) as build_client,
        patch.object(OpenAIAgent, "create", return_value=expected) as create,
    ):
        agent = create_agent("gpt-5.4", temperature=0.5)

    assert agent is expected
    build_client.assert_called_once_with("OpenAI")
    create.assert_called_once()
    assert create.call_args.kwargs["model"] == "gpt-5.4"
    assert create.call_args.kwargs["model_client"] is client
    assert create.call_args.kwargs["temperature"] == 0.5


@pytest.mark.parametrize("model_alias", ["uuid-openai", "GPT 5.4", "gpt-5.4"])
def test_create_agent_resolves_gateway_model_aliases(model_alias: str) -> None:
    expected = MagicMock()
    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=MODELS),
        patch("hud.agents.gateway.build_gateway_client", return_value=MagicMock()),
        patch.object(OpenAIAgent, "create", return_value=expected) as create,
    ):
        agent = create_agent(model_alias)

    assert agent is expected
    assert create.call_args.kwargs["model"] == "gpt-5.4"


def test_create_agent_shortcut_uses_gateway_provider() -> None:
    expected = MagicMock()
    with (
        patch("hud.agents.gateway.build_gateway_client", return_value=MagicMock()) as build_client,
        patch.object(ClaudeAgent, "create", return_value=expected),
    ):
        agent = create_agent("claude")

    assert agent is expected
    build_client.assert_called_once_with("anthropic")


def test_create_agent_openai_compatible_models_use_chat_agent_client() -> None:
    expected = MagicMock()
    client = MagicMock()
    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=MODELS),
        patch("hud.agents.gateway.build_gateway_client", return_value=client),
        patch.object(OpenAIChatAgent, "create", return_value=expected) as create,
    ):
        agent = create_agent("grok-4-1-fast")

    assert agent is expected
    assert create.call_args.kwargs["openai_client"] is client
    assert "model_client" not in create.call_args.kwargs


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("missing-model", "not found"),
        ("computer-use-preview", "Operator agent is no longer supported"),
        ("gemini-2.5-computer-use-preview", "Gemini CUA agent is no longer supported"),
    ],
)
def test_create_agent_rejects_unknown_or_stale_gateway_models(model: str, message: str) -> None:
    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=MODELS),
        pytest.raises(ValueError, match=message),
    ):
        create_agent(model)


def test_create_agent_rejects_gateway_model_with_invalid_agent_metadata() -> None:
    models = GatewayModelsResponse.model_validate(
        {
            "models": [
                {
                    "id": "bad-model",
                    "name": "Bad Model",
                    "model_name": "bad-model",
                    "provider": {"name": "OpenAI", "default_sdk_agent_type": None},
                }
            ]
        }
    ).models

    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=models),
        pytest.raises(ValueError, match="invalid agent type metadata"),
    ):
        create_agent("bad-model")


def test_create_agent_rejects_gateway_model_with_unknown_agent_metadata() -> None:
    models = GatewayModelsResponse.model_validate(
        {
            "models": [
                {
                    "id": "bad-model",
                    "name": "Bad Model",
                    "model_name": "bad-model",
                    "sdk_agent_type": "not_a_provider",
                    "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
                }
            ]
        }
    ).models

    with (
        patch("hud.agents.gateway._fetch_gateway_models", return_value=models),
        pytest.raises(ValueError, match="invalid agent type metadata"),
    ):
        create_agent("bad-model")


def _clear_gateway_model_cache() -> None:
    fetch_models = getattr(gateway_module, "_fetch_gateway_models")
    cache_clear = getattr(fetch_models, "cache_clear")
    cache_clear()


def test_create_agent_caches_gateway_model_lookup() -> None:
    response = MagicMock()
    response.json.return_value = {
        "models": [
            {
                "id": "model-id",
                "name": "Model",
                "model_name": "provider-model",
                "provider": {"name": "OpenAI", "default_sdk_agent_type": "openai"},
            }
        ]
    }
    expected = MagicMock()
    client = MagicMock()

    _clear_gateway_model_cache()
    try:
        with (
            patch("hud.agents.gateway.settings") as settings,
            patch("hud.agents.gateway.httpx.get", return_value=response) as get,
            patch("hud.agents.gateway.build_gateway_client", return_value=client),
            patch.object(OpenAIAgent, "create", return_value=expected) as create,
        ):
            settings.api_key = "hud-key"
            settings.hud_api_url = "https://api.example"

            first = create_agent("provider-model")
            second = create_agent("model-id")
    finally:
        _clear_gateway_model_cache()

    assert first is expected
    assert second is expected
    assert create.call_count == 2
    assert [call.kwargs["model"] for call in create.call_args_list] == [
        "provider-model",
        "provider-model",
    ]
    get.assert_called_once_with(
        "https://api.example/models/",
        headers={"Authorization": "Bearer hud-key"},
        timeout=10.0,
    )


def test_agent_type_config_and_gateway_metadata_do_not_import_optional_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    blocked = (
        "anthropic",
        "google.genai",
        "hud.agents.claude",
        "hud.agents.gemini",
    )

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if any(name == module or name.startswith(f"{module}.") for module in blocked):
            raise AssertionError(f"unexpected optional provider import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert AgentType.CLAUDE.config_cls().model_name == "Claude"
    assert AgentType.GEMINI.config_cls().model_name == "Gemini"
    assert AgentType.CLAUDE.gateway_provider == "anthropic"
    assert AgentType.GEMINI.gateway_provider == "gemini"


def test_build_gateway_client_uses_openai_compatible_client_by_default() -> None:
    with (
        patch("hud.agents.gateway.settings") as settings,
        patch("hud.agents.gateway.AsyncOpenAI") as client_cls,
    ):
        settings.api_key = "hud-key"
        settings.hud_gateway_url = "https://gateway.example"

        build_gateway_client("together")

    client_cls.assert_called_once_with(
        api_key="hud-key",
        base_url="https://gateway.example",
    )


def test_build_gateway_client_uses_anthropic_client_for_anthropic_provider() -> None:
    with (
        patch("hud.agents.gateway.settings") as settings,
        patch("anthropic.AsyncAnthropic") as client_cls,
    ):
        settings.api_key = "hud-key"
        settings.hud_gateway_url = "https://gateway.example"

        build_gateway_client("anthropic")

    client_cls.assert_called_once_with(
        api_key="hud-key",
        base_url="https://gateway.example",
    )


def test_build_gateway_client_uses_genai_client_for_gemini_provider() -> None:
    with (
        patch("hud.agents.gateway.settings") as settings,
        patch("google.genai.Client") as client_cls,
    ):
        settings.api_key = "hud-key"
        settings.hud_gateway_url = "https://gateway.example"

        build_gateway_client("gemini")

    client_cls.assert_called_once()
    assert client_cls.call_args.kwargs["api_key"] == "PLACEHOLDER"
    http_options = client_cls.call_args.kwargs["http_options"]
    assert http_options.api_version == "v1beta"
    assert http_options.base_url == "https://gateway.example"
    assert http_options.headers == {"Authorization": "Bearer hud-key"}
