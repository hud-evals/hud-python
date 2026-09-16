"""Tests for ``hud eval``."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hud.cli import eval as eval_mod
from hud.cli.__main__ import app
from hud.cli.eval import EvalConfig
from hud.eval import (
    Grade,
    HostedRuntime,
    HUDRuntime,
    Job,
    Run,
    Runtime,
    RuntimeConfig,
    Task,
    Taskset,
)
from hud.settings import settings
from hud.utils.exceptions import HudAuthenticationError
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from pathlib import Path

_TASKS_PY = """\
from hud import Environment

env = Environment("demo")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=0), solve(n=1)]
"""

_BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-profile"
_CONTAINER_ROW = '{"env": "demo", "id": "solve", "runtime_config": {"image": "example:latest"}}'


@dataclass
class _EvalCli:
    """``hud eval`` in a scratch project; ``Taskset.run`` is captured, not executed."""

    taskset: Taskset | None = None
    agent: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    job: Job = field(default_factory=lambda: Job(id="job-1", name="demo"))

    def invoke(self, *args: str, exit_code: int = 0) -> dict[str, Any]:
        result = CliRunner().invoke(app, ["eval", *args, "--json"])
        assert result.exit_code == exit_code, result.output
        return json.loads(result.stdout)


@pytest.fixture
def eval_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _EvalCli:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.py").write_text(_TASKS_PY, encoding="utf-8")
    monkeypatch.setattr(settings, "api_key", "sk-hud-test")
    for key in ("anthropic_api_key", "openai_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, key, None)
    cli = _EvalCli()

    async def fake_run(self: Taskset, agent: Any, **kwargs: Any) -> Job:
        cli.taskset, cli.agent, cli.kwargs = self, agent, kwargs
        return cli.job

    monkeypatch.setattr(Taskset, "run", fake_run)
    return cli


def _select_preset(model: str) -> Any:
    preset = next(p for p in eval_mod._AGENT_PRESETS if p.model == model)
    return lambda self, message, choices, **_: preset


# ─── config file contract ───────────────────────────────────────────────


def test_load_missing_returns_defaults_without_writing(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    assert EvalConfig.load(path) == EvalConfig()
    assert not path.exists()


def test_load_parses_eval_and_agent_sections(tmp_path: Path) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(
        '[eval]\nagent = "openai"\nmax_steps = 5\nruntime = "hosted"\n\n'
        '[openai]\nmodel = "gpt-4o"\n',
        encoding="utf-8",
    )
    cfg = EvalConfig.load(path)
    assert cfg.agent_type is not None and cfg.agent_type.value == "openai"
    assert cfg.max_steps == 5
    assert cfg.runtime == "hosted"
    assert cfg.agent_config == {"openai": {"model": "gpt-4o"}}


def test_load_resolves_env_var_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_EVAL_MODEL", "gpt-4o")
    path = tmp_path / ".hud_eval.toml"
    path.write_text('[openai]\nmodel = "${MY_EVAL_MODEL}"\n', encoding="utf-8")
    assert EvalConfig.load(path).agent_config["openai"]["model"] == "gpt-4o"


def test_load_rejects_unset_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUD_EVAL_TEST_UNSET", raising=False)
    path = tmp_path / ".hud_eval.toml"
    path.write_text('[openai]\nmodel = "${HUD_EVAL_TEST_UNSET}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$\{HUD_EVAL_TEST_UNSET\} is not set"):
        EvalConfig.load(path)


@pytest.mark.parametrize(
    "contents, match",
    [
        ('[eval]\nmodle = "x"\n', "modle"),
        ('[eval]\nagent = "not-an-agent"\n', "claude"),
        ('[eval]\nruntime = "cloud"\n', "Unknown runtime"),
        ("[other]\nx = 1\n", "unknown sections: other"),
    ],
)
def test_load_rejects_invalid_configuration(tmp_path: Path, contents: str, match: str) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        EvalConfig.load(path)


def test_placement_defaults_from_source(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    assert EvalConfig(source=str(tasks)).with_placement().runtime == "local"
    assert EvalConfig(source="My Tasks").with_placement().runtime == "hosted"
    assert EvalConfig(source="My Tasks", runtime="hud").with_placement().runtime == "hud"
    with pytest.raises(ValueError, match="platform taskset with no env source"):
        EvalConfig(source="My Tasks", runtime="local").with_placement()


@pytest.mark.parametrize("fields", [{"runtime": "hud"}, {"runtime": "hosted"}, {"gateway": True}])
def test_platform_features_require_hud_key(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]
) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    with pytest.raises(HudAuthenticationError):
        EvalConfig(agent_type="gemini", **fields).require_credentials()


def test_openai_compatible_requires_a_model() -> None:
    with pytest.raises(ValueError, match="Model name is required"):
        EvalConfig(agent_type="openai_compatible").require_credentials()


def test_agent_kwargs_model_precedence_and_aliases() -> None:
    cfg = EvalConfig(
        agent_type="openai_compatible",
        model="glm-5.2",
        max_steps=7,
        auto_respond=True,
        agent_config={"openai_compatible": {"temperature": 0.5, "model": "kimi-2.6"}},
    )
    assert cfg.agent_kwargs() == {
        "temperature": 0.5,
        "model": "z-ai/glm-5.2",
        "max_steps": 7,
        "auto_respond": True,
    }
    from_section = EvalConfig(
        agent_type="openai_compatible", agent_config={"openai_compatible": {"model": "kimi-2.6"}}
    )
    assert from_section.agent_kwargs()["model"] == "moonshotai/kimi-k2.6"


# ─── command: planning and validation ──────────────────────────────────


@pytest.mark.parametrize("args", [[], ["tasks.json", "claude"]])
def test_dry_run_does_not_prompt_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.json").write_text("[]")
    monkeypatch.setattr(HUDConsole, "select", lambda *a, **k: pytest.fail("dry-run prompted"))
    result = CliRunner().invoke(app, ["eval", *args, "--dry-run", "--json"])
    assert result.exit_code == (0 if args else 2), result.output
    payload = json.loads(result.stdout)
    if args:
        assert payload["runtime"] == "local"
        assert payload["remote"] is False
        assert payload["agent"] == "claude"
    else:
        assert payload["error"] == "usage"
    assert not (tmp_path / ".hud_eval.toml").exists()


@pytest.mark.parametrize("contents", ["[eval", '[eval]\nagent="invalid"\n'])
def test_invalid_configuration_is_a_structured_error(tmp_path, monkeypatch, contents) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / ".hud_eval.toml"
    config.write_text(contents)
    result = CliRunner().invoke(app, ["eval", "tasks.json", "openai", "--dry-run", "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "usage"
    assert config.read_text() == contents


def test_full_expands_to_all_auto_respond_and_100_steps(eval_cli: _EvalCli) -> None:
    payload = eval_cli.invoke("tasks.py", "openai", "--full", "--dry-run")
    assert payload["all"] is True
    assert payload["max_steps"] == 100
    eval_cli.invoke("tasks.py", "openai", "--full", "--yes")
    assert eval_cli.agent.config.auto_respond is True
    assert eval_cli.agent.config.max_steps == 100


def test_runtime_and_remote_flags_conflict(eval_cli: _EvalCli) -> None:
    payload = eval_cli.invoke("tasks.py", "openai", "--runtime", "hud", "--remote", exit_code=2)
    assert payload["error"] == "usage"
    assert "mutually exclusive" in payload["message"]


def test_config_flag_lands_in_agent_config(eval_cli: _EvalCli) -> None:
    eval_cli.invoke("tasks.py", "claude", "--config", "max_tokens=100", "--yes")
    assert eval_cli.agent.config.max_tokens == 100
    eval_cli.invoke("tasks.py", "claude", "-c", "claude.max_tokens=200", "--yes")
    assert eval_cli.agent.config.max_tokens == 200


@pytest.mark.parametrize(
    "args, match",
    [
        (["tasks.py", "claude", "--config", "max_tokens"], "key=value"),
        (["tasks.py", "--config", "max_tokens=1"], "needs an agent"),
        (["tasks.py", "claude", "--config", "robot.max_tokens=1"], "not a valid AgentType"),
    ],
)
def test_malformed_config_flag_is_a_usage_error(
    eval_cli: _EvalCli, args: list[str], match: str
) -> None:
    payload = eval_cli.invoke(*args, "--yes", exit_code=2)
    assert payload["error"] == "usage"
    assert match in payload["message"]


def test_gateway_model_alias_selects_agent_and_model(eval_cli: _EvalCli, monkeypatch) -> None:
    from hud.utils.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id="z-ai/glm-5.2",
        model_name="z-ai/glm-5.2",
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.agents.list_gateway_models", lambda: [model])
    eval_cli.invoke("tasks.py", "glm-5.2", "--yes")
    assert type(eval_cli.agent).__name__ == "OpenAIChatAgent"
    assert eval_cli.agent.config.model == "z-ai/glm-5.2"


# ─── command: task selection and placement ─────────────────────────────


def test_default_runs_only_the_first_task(eval_cli: _EvalCli) -> None:
    eval_cli.invoke("tasks.py", "openai", "--yes")
    assert eval_cli.taskset is not None
    assert [task.args for task in eval_cli.taskset] == [{"n": 0}]
    eval_cli.invoke("tasks.py", "openai", "--all", "--yes")
    assert eval_cli.taskset is not None and len(eval_cli.taskset) == 2
    eval_cli.invoke("tasks.py", "openai", "--task-ids", "1", "--yes")
    assert eval_cli.taskset is not None
    assert [task.args for task in eval_cli.taskset] == [{"n": 1}]


def test_unknown_task_ids_fail(eval_cli: _EvalCli) -> None:
    payload = eval_cli.invoke("tasks.py", "openai", "--task-ids", "nope", "--yes", exit_code=2)
    assert "No tasks matching: nope" in payload["message"]


def test_group_and_concurrency_reach_the_scheduler(eval_cli: _EvalCli) -> None:
    eval_cli.invoke("tasks.py", "openai", "--group", "3", "--max-concurrent", "2", "--yes")
    assert eval_cli.kwargs["group"] == 3
    assert eval_cli.kwargs["max_concurrent"] == 2


def test_local_placement_routes_each_row(eval_cli: _EvalCli, tmp_path: Path, monkeypatch) -> None:
    docker = MagicMock(name="docker")
    subprocess = MagicMock(name="subprocess")
    monkeypatch.setattr(eval_mod, "DockerRuntime", lambda: docker)
    monkeypatch.setattr(eval_mod, "SubprocessRuntime", lambda env: subprocess)
    image = Task(env="image", id="run", runtime_config=RuntimeConfig(image="example:latest"))
    (tmp_path / "mixed.py").write_text(
        _TASKS_PY + "from hud.eval import RuntimeConfig, Task\n"
        "tasks.append(Task(env='image', id='run', "
        "runtime_config=RuntimeConfig(image='example:latest')))\n",
        encoding="utf-8",
    )
    try:
        eval_cli.invoke("mixed.py", "openai", "--all", "--yes")
    finally:
        sys.modules.pop("mixed", None)
    placement = eval_cli.kwargs["runtime"]
    assert eval_cli.taskset is not None
    bound = next(iter(eval_cli.taskset))

    assert placement(bound) is subprocess.return_value
    assert placement(image) is docker.return_value


def test_portable_rows_are_refused_before_running(eval_cli: _EvalCli, tmp_path: Path) -> None:
    (tmp_path / "rows.json").write_text(
        '[{"env": "demo", "id": "a"}, {"env": "demo", "id": "b"}]', encoding="utf-8"
    )
    payload = eval_cli.invoke("rows.json", "openai", "--all", "--yes", exit_code=2)
    assert payload["error"] == "usage"
    assert "2 task(s) have no bound Environment or container image (a, b)" in payload["message"]
    assert "--runtime hud" in payload["message"]
    assert eval_cli.taskset is None  # refused before Taskset.run


def test_explicit_placements(eval_cli: _EvalCli) -> None:
    eval_cli.invoke("tasks.py", "openai", "--runtime", "hud", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HUDRuntime)
    eval_cli.invoke("tasks.py", "openai", "--remote", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HostedRuntime)
    eval_cli.invoke("tasks.py", "openai", "--runtime", "hosted", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HostedRuntime)
    eval_cli.invoke("tasks.py", "openai", "--runtime", "tcp://127.0.0.1:7000", "--yes")
    assert eval_cli.kwargs["runtime"] == Runtime("tcp://127.0.0.1:7000")


def test_python_task_source_loads_on_main_thread(eval_cli: _EvalCli, tmp_path: Path) -> None:
    marker = tmp_path / "main-thread.txt"
    (tmp_path / "probe.py").write_text(
        "import threading\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text("
        "str(threading.current_thread() is threading.main_thread()))\n"
        "tasks = []\n",
        encoding="utf-8",
    )
    try:
        payload = eval_cli.invoke("probe.py", "openai", "--yes", exit_code=2)
    finally:
        sys.modules.pop("probe", None)
    assert "No runnable Tasks" in payload["message"]
    assert marker.read_text(encoding="utf-8") == "True"


# ─── command: agent construction ───────────────────────────────────────


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
def test_provider_key_wins_unless_gateway_is_forced(
    eval_cli: _EvalCli,
    monkeypatch,
    agent_type,
    key_attr,
    factory,
    client_attr,
    force_gateway,
    provider_key,
) -> None:
    monkeypatch.setattr(settings, key_attr, provider_key)
    direct = MagicMock(return_value=object())
    gateway = MagicMock(return_value=object())
    monkeypatch.setattr(factory, direct)
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", gateway)
    flags = ["--gateway"] if force_gateway else []
    eval_cli.invoke("tasks.py", agent_type, *flags, "--yes")
    if provider_key and not force_gateway:
        direct.assert_called_once_with(api_key=provider_key)
        gateway.assert_not_called()
        assert getattr(eval_cli.agent, client_attr) is direct.return_value
    else:
        gateway.assert_called_once()
        direct.assert_not_called()
        assert getattr(eval_cli.agent, client_attr) is gateway.return_value


def test_hosted_agent_keeps_client_out_of_serialized_config(
    eval_cli: _EvalCli, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "provider-key")
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", MagicMock(return_value=object()))
    eval_cli.invoke("tasks.py", "openai", "--remote", "--yes")
    assert eval_cli.agent.config.model_client is None
    assert "model_client" not in eval_cli.agent.hosted_spec()["config"]


def test_openai_compatible_routes_through_gateway_despite_openai_key(
    eval_cli: _EvalCli, monkeypatch
) -> None:
    """A third-party chat model is not an OpenAI model: OPENAI_API_KEY must not claim it."""
    monkeypatch.setattr(settings, "openai_api_key", "provider-key")
    gateway = MagicMock(return_value=object())
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", gateway)
    eval_cli.invoke("tasks.py", "openai_compatible", "--model", "MiniMax-M3", "--yes")
    gateway.assert_called_once_with("openai")
    assert eval_cli.agent.oai is gateway.return_value
    assert eval_cli.agent.config.base_url is None


def test_openai_compatible_custom_endpoint_is_used_directly(
    eval_cli: _EvalCli, monkeypatch
) -> None:
    client = MagicMock(return_value=object())
    monkeypatch.setattr("hud.agents.openai_compatible.agent.AsyncOpenAI", client)
    eval_cli.invoke(
        "tasks.py",
        "openai_compatible",
        "-m",
        "custom",
        "-c",
        "api_key=custom-key",
        "-c",
        "base_url=https://custom.example",
        "--gateway",
        "--yes",
    )
    client.assert_called_once_with(api_key="custom-key", base_url="https://custom.example")
    assert eval_cli.agent.oai is client.return_value


def test_bedrock_arn_in_config_selects_bedrock_client(eval_cli: _EvalCli, monkeypatch) -> None:
    monkeypatch.setattr(settings, "aws_access_key_id", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_access_key", "secret")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    bedrock = MagicMock(return_value=object())
    monkeypatch.setattr("anthropic.AsyncAnthropicBedrock", bedrock)
    eval_cli.invoke("tasks.py", "claude", "--config", f"checkpoint_name={_BEDROCK_ARN}", "--yes")
    assert eval_cli.agent.config.model == _BEDROCK_ARN
    assert eval_cli.agent.config.model_client is bedrock.return_value


def test_interactive_preset_selects_agent_and_model(eval_cli: _EvalCli, monkeypatch) -> None:
    monkeypatch.setattr(HUDConsole, "select", _select_preset("MiniMax-M3"))
    eval_cli.invoke("tasks.py", "--yes")
    assert eval_cli.agent.config.model == "MiniMax-M3"
    assert eval_cli.agent.config.model_name == "MiniMax M3"


# ─── command: source discovery and results ─────────────────────────────


def test_missing_source_with_no_tasks_files_is_not_found(eval_cli: _EvalCli, monkeypatch) -> None:
    monkeypatch.setattr(HUDConsole, "select", lambda *a, **k: pytest.fail("prompted"))
    payload = eval_cli.invoke("--yes", exit_code=1)
    assert payload["error"] == "not_found"


def test_single_tasks_file_is_picked_without_prompting(
    eval_cli: _EvalCli, tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "rows.json").write_text(f"[{_CONTAINER_ROW}]", encoding="utf-8")
    (tmp_path / ".hidden.json").write_text("[]", encoding="utf-8")

    def select(self: HUDConsole, message: str, choices: Any, **_: Any) -> Any:
        assert message == "Select an agent:"
        return next(p for p in eval_mod._AGENT_PRESETS if p.model == "gpt-5.6")

    monkeypatch.setattr(HUDConsole, "select", select)
    payload = eval_cli.invoke("--yes")
    assert payload["source"] == "rows.json"


def test_several_tasks_files_prompt_for_one(
    eval_cli: _EvalCli, tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "rows.json").write_text(f"[{_CONTAINER_ROW}]", encoding="utf-8")
    (tmp_path / "more.jsonl").write_text(_CONTAINER_ROW + "\n", encoding="utf-8")
    seen: list[str] = []

    def select(self: HUDConsole, message: str, choices: Any, **_: Any) -> str:
        assert message == "Select a tasks file"
        seen.extend(choices)
        return "more.jsonl"

    monkeypatch.setattr(HUDConsole, "select", select)
    # The agent has to come from config: without a source there is no second positional.
    (tmp_path / ".hud_eval.toml").write_text('[eval]\nagent = "openai"\n', encoding="utf-8")
    payload = eval_cli.invoke("--yes")
    assert payload["source"] == "more.jsonl"
    assert seen == ["more.jsonl", "rows.json"]


def test_result_payload_uses_job_metrics(eval_cli: _EvalCli) -> None:
    graded = Run(None, "solve", {})
    graded.grade = Grade(reward=1.0)
    graded.slug = "solve"
    errored = Run(None, "solve", {})
    errored.grade = Grade(is_error=True)
    eval_cli.job.runs.extend([graded, errored])

    payload = eval_cli.invoke("tasks.py", "openai", "--yes")

    assert payload["job_id"] == "job-1"
    assert payload["run_count"] == 2
    assert payload["mean_reward"] == 1.0
    assert payload["error_count"] == 1
    assert [run["slug"] for run in payload["runs"]] == ["solve", None]
    assert [run["is_error"] for run in payload["runs"]] == [False, False]
