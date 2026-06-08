"""``hud.cli.eval.EvalConfig`` — agent parsing, kwargs building, TOML load, CLI merge.

Pure config logic; no agent is constructed and no network is touched.
"""
# pyright: reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli import eval_config as eval_mod
from hud.cli.eval_config import EvalConfig, _is_bedrock_arn

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


def test_display_renders() -> None:
    EvalConfig(agent_type="openai", model="gpt").display()
