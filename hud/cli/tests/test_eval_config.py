"""``hud.cli.eval.EvalConfig`` — agent parsing, kwargs building, TOML load, CLI merge.

Pure config logic; no agent is constructed and no network is touched.
"""
# pyright: reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli import eval as eval_mod
from hud.cli.eval import EvalConfig, _is_bedrock_arn

if TYPE_CHECKING:
    from pathlib import Path

_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"


def test_is_bedrock_arn() -> None:
    assert _is_bedrock_arn(_ARN) is True
    assert _is_bedrock_arn("claude-sonnet-4-6") is False
    assert _is_bedrock_arn(None) is False


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


def test_get_agent_kwargs_requires_agent_type() -> None:
    with pytest.raises(ValueError, match="agent_type must be set"):
        EvalConfig().get_agent_kwargs()


def test_validate_api_keys_noop_without_agent() -> None:
    EvalConfig().validate_api_keys()  # no agent -> returns without error


def test_validate_api_keys_openai_compatible_requires_model() -> None:
    cfg = EvalConfig(agent_type="openai_compatible")
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_remote_needs_only_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted placement: no provider key required, and --gateway is dropped
    (a local gateway model_client could not travel with the submission)."""
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True, gateway=True)
    cfg.validate_api_keys()
    assert cfg.gateway is False


def test_validate_api_keys_remote_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", remote=True)
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_hud_runtime_requires_hud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    with pytest.raises(typer.Exit):
        cfg.validate_api_keys()


def test_validate_api_keys_hud_runtime_keeps_local_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    cfg = EvalConfig(agent_type="gemini", runtime="hud")
    cfg.validate_api_keys()
    assert cfg.gateway is True


def test_resolve_placement_runtime_hud_uses_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HUDRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(runtime="hud"), tmp_path)

    assert isinstance(placement, HUDRuntime)


def test_resolve_placement_remote_uses_hosted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.eval import HostedRuntime
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "sk-hud-test")

    placement = eval_mod._resolve_placement(EvalConfig(remote=True), tmp_path)

    assert isinstance(placement, HostedRuntime)


def test_runtime_cli_override_clears_config_remote() -> None:
    cfg = EvalConfig(remote=True).merge_cli(runtime="hud")

    assert cfg.runtime == "hud"
    assert cfg.remote is False


def test_runtime_cli_rejects_remote_flag_conflict() -> None:
    with pytest.raises(ValueError, match="--runtime and --remote are mutually exclusive"):
        EvalConfig().merge_cli(runtime="hud", remote=True)


def test_load_missing_writes_template(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    cfg = EvalConfig.load(str(path))
    assert path.exists()  # template generated
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
    with pytest.raises(typer.Exit):
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
