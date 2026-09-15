"""Tests for ``hud.cli.eval``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hud.cli import eval as eval_mod
from hud.cli.eval import (
    EvalConfig,
    _build_agent,
    find_tasks_file,
)
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError


def test_parse_agent_type_accepts_known_value() -> None:
    cfg = EvalConfig(agent_type="openai")
    assert cfg.agent_type is not None
    assert cfg.agent_type.value == "openai"


def test_parse_agent_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid agent"):
        EvalConfig(agent_type="not-an-agent")


def test_get_agent_kwargs_model_precedence_and_flags() -> None:
    cfg = EvalConfig(
        agent_type="openai",
        model="gpt-cli",
        verbose=True,
        agent_config={"openai": {"temperature": 0.5, "model": "gpt-config"}},
    )
    kwargs = cfg.get_agent_kwargs()
    assert kwargs["model"] == "gpt-cli"  # CLI model wins over config model
    assert kwargs["temperature"] == 0.5
    assert kwargs["verbose"] is True


def test_get_agent_kwargs_normalizes_gateway_model_alias() -> None:
    cfg = EvalConfig(agent_type="openai_compatible", model="glm-5.2")

    assert cfg.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_get_agent_kwargs_normalizes_config_model_alias() -> None:
    cfg = EvalConfig(
        agent_type="openai_compatible",
        agent_config={"openai_compatible": {"model": "glm-5.2"}},
    )

    assert cfg.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_get_agent_kwargs_requires_agent_type() -> None:
    with pytest.raises(ValueError, match="agent_type must be set"):
        EvalConfig().get_agent_kwargs()


def test_validate_api_keys_noop_without_agent() -> None:
    EvalConfig().validate_api_keys()  # no agent -> returns without error


def test_validate_api_keys_openai_compatible_requires_model() -> None:
    cfg = EvalConfig(agent_type="openai_compatible")
    with pytest.raises(ValueError):
        cfg.validate_api_keys()


def test_validate_api_keys_remote_needs_only_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted placement needs no local provider credentials."""
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True, gateway=True)
    cfg.validate_api_keys()
    assert cfg.gateway is True


def test_validate_api_keys_remote_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True)
    with pytest.raises(HudAuthenticationError):
        cfg.validate_api_keys()


def test_validate_api_keys_hud_runtime_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    with pytest.raises(HudAuthenticationError):
        cfg.validate_api_keys()


def test_validate_api_keys_preserves_explicit_routing_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    cfg.validate_api_keys()
    assert cfg.gateway is False


def test_resolve_placement_runtime_hud_uses_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HUDRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(runtime="hud"), tmp_path, [])

    assert isinstance(placement, HUDRuntime)


def test_resolve_placement_remote_uses_hosted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HostedRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(remote=True), tmp_path, [])

    assert isinstance(placement, HostedRuntime)


def test_resolve_placement_routes_each_local_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hud.eval as eval_api
    from hud.environment import Environment
    from hud.eval import RuntimeConfig, Task

    env = Environment("source")

    @env.template(id="run")
    async def run() -> None:
        yield "prompt"
        yield 1.0

    docker = lambda task: ("docker", task)
    subprocess = lambda task: ("subprocess", task)
    monkeypatch.setattr(eval_api, "DockerRuntime", lambda: docker)
    monkeypatch.setattr(eval_api, "SubprocessRuntime", lambda _path: subprocess)

    bound = run()
    image = Task(env="image", id="run", runtime_config=RuntimeConfig(image="example:latest"))
    placement = eval_mod._resolve_placement(
        EvalConfig(runtime="local"),
        tmp_path,
        [bound, image],
    )

    assert placement(bound) == ("subprocess", bound)
    assert placement(image) == ("docker", image)


def test_runtime_cli_override_clears_config_remote() -> None:
    cfg = EvalConfig(remote=True).merge_cli(runtime="hud")

    assert cfg.runtime == "hud"
    assert cfg.remote is False


def test_runtime_cli_rejects_remote_flag_conflict() -> None:
    with pytest.raises(ValueError, match="--runtime and --remote are mutually exclusive"):
        EvalConfig().merge_cli(runtime="hud", remote=True)


def test_load_missing_returns_defaults_without_writing(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    cfg = EvalConfig.load(str(path))
    assert not path.exists()
    assert isinstance(cfg, EvalConfig)


def test_load_parses_sections(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(
        '[eval]\nagent = "openai"\nmax_steps = 5\n\n[openai]\nmodel = "gpt-4o"\n',
        encoding="utf-8",
    )
    cfg = EvalConfig.load(str(path))
    assert cfg.agent_type is not None and cfg.agent_type.value == "openai"
    assert cfg.max_steps == 5
    assert cfg.agent_config["openai"]["model"] == "gpt-4o"


def test_load_resolves_env_var_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_EVAL_MODEL", "gpt-4o")
    path = tmp_path / ".hud_eval.toml"
    path.write_text(
        '[eval]\nagent = "openai"\n\n[openai]\nmodel = "${MY_EVAL_MODEL}"\n',
        encoding="utf-8",
    )
    cfg = EvalConfig.load(str(path))
    assert cfg.agent_config["openai"]["model"] == "gpt-4o"


def test_merge_cli_overrides_fields() -> None:
    merged = EvalConfig().merge_cli(agent="openai", task_ids="a, b", max_steps=7)
    assert merged.agent_type is not None and merged.agent_type.value == "openai"
    assert merged.task_ids == ["a", "b"]
    assert merged.max_steps == 7


def test_merge_cli_resolves_gateway_model_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.utils.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id="z-ai/glm-5.2",
        model_name="z-ai/glm-5.2",
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.agents.list_gateway_models", lambda: [model])

    merged = EvalConfig().merge_cli(agent="glm-5.2")

    assert merged.agent_type is not None and merged.agent_type.value == "openai_compatible"
    assert merged.model == "z-ai/glm-5.2"


def test_merge_cli_config_model_alias_is_normalized() -> None:
    merged = EvalConfig(agent_type="openai_compatible").merge_cli(
        config=["openai_compatible.model=glm-5.2"]
    )

    assert merged.get_agent_kwargs()["model"] == "z-ai/glm-5.2"


def test_merge_cli_namespaced_config() -> None:
    merged = EvalConfig().merge_cli(config=["claude.max_tokens=100"])
    assert merged.agent_config["claude"]["max_tokens"] == 100


def test_resolve_agent_interactive_uses_selected_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    preset = eval_mod._AGENT_PRESETS[0]
    monkeypatch.setattr(eval_mod.hud_console, "select", lambda *a, **k: preset)
    resolved = EvalConfig().resolve_agent_interactive()
    assert resolved.agent_type == preset.agent_type


def test_resolve_runtime_local_file_defaults_to_local(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    cfg = EvalConfig(source=str(tasks)).resolve_runtime()
    assert cfg.runtime == "local"


async def test_python_task_source_loads_on_main_thread(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tasks.py"
    marker = tmp_path / "main-thread.txt"
    source.write_text(
        "import threading\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text("
        "str(threading.current_thread() is threading.main_thread()))\n"
        "tasks = []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        await eval_mod._run_evaluation(EvalConfig(source=str(source), agent_type="openai"))

    assert marker.read_text(encoding="utf-8") == "True"


def test_resolve_runtime_slug_defaults_to_remote() -> None:
    cfg = EvalConfig(source="My Tasks").resolve_runtime()
    assert cfg.runtime is None
    assert cfg.remote is True


def test_resolve_runtime_explicit_runtime_is_honored() -> None:
    cfg = EvalConfig(source="My Tasks", runtime="hud").resolve_runtime()
    assert cfg.runtime == "hud"
    cfg = EvalConfig(source="My Tasks", runtime="tcp://127.0.0.1:7000").resolve_runtime()
    assert cfg.runtime == "tcp://127.0.0.1:7000"


def test_resolve_runtime_local_against_slug_errors() -> None:
    cfg = EvalConfig(source="My Tasks", runtime="local")
    with pytest.raises(ValueError):
        cfg.resolve_runtime()


def test_display_renders() -> None:
    EvalConfig(agent_type="openai", model="gpt").display()


def test_eval_max_steps_lands_in_agent_config() -> None:
    cfg = EvalConfig(
        source="tasks.py",
        agent_type="openai",
        max_steps=17,
        agent_config={"openai": {"model_client": object()}},
    )
    agent = eval_mod._build_agent(cfg)
    assert agent.config.max_steps == 17


@pytest.mark.parametrize("args", [[], ["tasks.json", "claude"]])
def test_eval_dry_run_does_not_prompt_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    import json

    from typer.testing import CliRunner

    from hud.cli.app import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.json").write_text("[]")
    from hud.utils.hud_console import HUDConsole

    monkeypatch.setattr(HUDConsole, "select", lambda *a, **k: pytest.fail("dry-run prompted"))
    result = CliRunner().invoke(app, ["eval", *args, "--dry-run", "--json"])
    assert result.exit_code == (0 if args else 2), result.output
    payload = json.loads(result.stdout)
    if args:
        assert payload["runtime"] == "local"
    else:
        assert payload["error"] == "usage"
    assert not (tmp_path / ".hud_eval.toml").exists()


@pytest.mark.parametrize(
    "agent_type, key_attr, factory, client_attr",
    [
        ("openai", "openai_api_key", "hud.utils.gateway.AsyncOpenAI", "openai_client"),
        ("claude", "anthropic_api_key", "anthropic.AsyncAnthropic", "anthropic_client"),
        ("gemini", "gemini_api_key", "google.genai.Client", "gemini_client"),
    ],
)
@pytest.mark.parametrize(
    "force_gateway, provider_key", [(False, "provider-key"), (False, None), (True, "provider-key")]
)
def test_eval_provider_preference_and_explicit_gateway(
    monkeypatch, agent_type, key_attr, factory, client_attr, force_gateway, provider_key
):
    from unittest.mock import MagicMock

    monkeypatch.setattr("hud.settings.settings.api_key", "hud-key")
    monkeypatch.setattr(f"hud.settings.settings.{key_attr}", provider_key)
    direct = MagicMock(return_value=object())
    gateway = MagicMock(return_value=object())
    monkeypatch.setattr(factory, direct)
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", gateway)
    cfg = EvalConfig(agent_type=agent_type, gateway=force_gateway)
    assert cfg.agent_type is not None
    cfg.validate_api_keys()
    agent = eval_mod._build_agent(cfg)
    if provider_key and not force_gateway:
        direct.assert_called_once_with(api_key=provider_key)
        gateway.assert_not_called()
        assert getattr(agent, client_attr) is direct.return_value
    else:
        gateway.assert_called_once_with(cfg.agent_type.gateway_provider)
        direct.assert_not_called()
        assert getattr(agent, client_attr) is gateway.return_value


def test_eval_hosted_agent_keeps_client_out_of_serialized_config(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr("hud.settings.settings.api_key", "hud-key")
    monkeypatch.setattr("hud.settings.settings.openai_api_key", "provider-key")
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", MagicMock(return_value=object()))
    cfg = EvalConfig(agent_type="openai", remote=True, gateway=True)
    cfg.validate_api_keys()
    agent = eval_mod._build_agent(cfg)
    assert agent.config.model_client is None
    assert "model_client" not in agent.hosted_spec()["config"]


@pytest.mark.parametrize("contents", ["[eval", '[eval]\nagent="invalid"\n'])
def test_eval_invalid_configuration_is_a_structured_error(tmp_path, monkeypatch, contents):
    import json

    from typer.testing import CliRunner

    from hud.cli.app import app

    monkeypatch.chdir(tmp_path)
    config = tmp_path / ".hud_eval.toml"
    config.write_text(contents)
    result = CliRunner().invoke(app, ["eval", "tasks.json", "openai", "--dry-run", "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "usage"
    assert config.read_text() == contents


def test_eval_custom_endpoint_overrides_gateway_selection(monkeypatch):
    from unittest.mock import MagicMock

    client = MagicMock(return_value=object())
    monkeypatch.setattr("hud.settings.settings.api_key", "hud-key")
    monkeypatch.setattr("hud.agents.openai_compatible.agent.AsyncOpenAI", client)
    cfg = EvalConfig(
        agent_type="openai_compatible",
        model="custom",
        gateway=True,
        agent_config={
            "openai_compatible": {"api_key": "custom-key", "base_url": "https://custom.example"}
        },
    )
    cfg.validate_api_keys()
    agent = eval_mod._build_agent(cfg)
    client.assert_called_once_with(api_key="custom-key", base_url="https://custom.example")
    assert agent.oai is client.return_value


def test_eval_openai_compatible_routes_through_gateway_despite_openai_key(monkeypatch):
    """A third-party chat model is not an OpenAI model: OPENAI_API_KEY must not claim it."""
    gateway = MagicMock(return_value=object())
    monkeypatch.setattr("hud.settings.settings.api_key", "hud-key")
    monkeypatch.setattr("hud.settings.settings.openai_api_key", "provider-key")
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", gateway)
    preset = next(p for p in eval_mod._AGENT_PRESETS if p.model == "MiniMax-M3")
    monkeypatch.setattr(eval_mod.hud_console, "select", lambda *a, **k: preset)
    cfg = EvalConfig().resolve_agent_interactive()
    agent = eval_mod._build_agent(cfg)
    gateway.assert_called_once_with("openai")
    assert agent.oai is gateway.return_value
    assert agent.config.model == "MiniMax-M3"
    assert agent.config.base_url is None


@patch("pathlib.Path.cwd")
def test_find_tasks_file_with_arg(mock_cwd):
    assert find_tasks_file("some/path.json") == "some/path.json"
    mock_cwd.assert_not_called()


@patch("pathlib.Path.cwd")
def test_find_tasks_file_no_files(mock_cwd):
    mock_path = MagicMock(spec=Path)
    mock_path.glob.return_value = []
    mock_cwd.return_value = mock_path

    with pytest.raises(FileNotFoundError, match="No task JSON or JSONL files found"):
        find_tasks_file(None)


@patch("hud.cli.eval.hud_console")
@patch("pathlib.Path.cwd")
def test_find_tasks_file_single_file(mock_cwd, mock_console):
    mock_path = MagicMock(spec=Path)
    mock_file = MagicMock(spec=Path)
    mock_file.name = "test.json"

    def glob_side_effect(pattern):
        if pattern == "*.json":
            return [mock_file]
        return []

    mock_path.glob.side_effect = glob_side_effect
    mock_cwd.return_value = mock_path

    result = find_tasks_file(None)
    assert result == "test.json"
    mock_console.select.assert_not_called()


@patch("hud.cli.eval.hud_console")
@patch("pathlib.Path.cwd")
def test_find_tasks_file_multiple_files(mock_cwd, mock_console):
    mock_path = MagicMock(spec=Path)
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = "test1.json"
    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = "test2.jsonl"

    def glob_side_effect(pattern):
        if pattern == "*.json":
            return [mock_file1]
        if pattern == "*.jsonl":
            return [mock_file2]
        return []

    mock_path.glob.side_effect = glob_side_effect
    mock_cwd.return_value = mock_path
    mock_console.select.return_value = "test2.jsonl"

    result = find_tasks_file(None)

    assert result == "test2.jsonl"
    mock_console.select.assert_called_once()
    call_args = mock_console.select.call_args
    assert call_args[0][0] == "Select a tasks file"
    assert "test1.json" in call_args[1]["choices"]
    assert "test2.jsonl" in call_args[1]["choices"]


def test_build_agent_detects_bedrock_arn_from_config_checkpoint_name() -> None:
    """Regression: ARN in [claude].checkpoint_name should trigger Bedrock client."""
    arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile"
    cfg = EvalConfig(
        agent_type=AgentType.CLAUDE,
        model=None,  # no CLI --model
        agent_config={"claude": {"checkpoint_name": arn}},
    )

    with (
        patch("hud.settings.settings.aws_access_key_id", "AKIATEST"),
        patch("hud.settings.settings.aws_secret_access_key", "secret"),
        patch("hud.settings.settings.aws_region", "us-east-1"),
        patch("anthropic.AsyncAnthropicBedrock", return_value=MagicMock()) as mock_bedrock,
    ):
        agent = _build_agent(cfg)

    assert agent.config.model == arn
    assert agent.config.model_client is mock_bedrock.return_value
    mock_bedrock.assert_called_once()
