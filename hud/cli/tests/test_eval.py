"""Tests for ``hud eval``."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hud.agents import OpenAIAgent
from hud.agents.types import ClaudeConfig
from hud.cli import eval as eval_mod
from hud.cli.__main__ import app
from hud.cli.eval import EvalConfig
from hud.eval import (
    DaytonaRuntime,
    DockerRuntime,
    Grade,
    HostedRuntime,
    HUDRuntime,
    Job,
    ModalRuntime,
    Run,
    Runtime,
    RuntimeConfig,
    Task,
    Taskset,
)
from hud.settings import settings
from hud.utils.gateway import GatewayModelInfo
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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


def _catalog(*rows: tuple[str, str, str, str]) -> list[GatewayModelInfo]:
    """(sdk_agent_type, name, model_name, created_at) rows, as the gateway catalog returns them."""
    return [
        GatewayModelInfo(
            id=model_name,
            name=name,
            model_name=model_name,
            sdk_agent_type=agent,
            created_at=created,
        )
        for agent, name, model_name, created in rows
    ]


def _picker(agent: str, pick_model: Callable[[list[Any]], Any]) -> Any:
    """A ``HUDConsole.select`` that answers the agent prompt with ``agent`` and the model
    prompt via ``pick_model(choices)``."""

    def select(self: HUDConsole, message: str, choices: Any, **_: Any) -> Any:
        if message == "Select an agent:":
            return agent
        assert message == "Select a model:"
        return pick_model(list(choices))

    return select


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


@pytest.mark.parametrize(
    "contents, source, flags, expected",
    [
        ("remote = true", "tasks.py", [], "hosted"),
        ("remote = false", "tasks.py", [], "local"),
        ("remote = false", "My Tasks", [], "hosted"),
        ('remote = true\nruntime = "local"', "tasks.py", [], "local"),
        ('remote = false\nruntime = "hud"', "tasks.py", [], "hud"),
        ("remote = true", "tasks.py", ["--runtime", "local"], "local"),
        ("remote = false", "tasks.py", ["--remote"], "hosted"),
    ],
)
def test_legacy_remote_config_resolves_placement(
    eval_cli: _EvalCli,
    tmp_path: Path,
    contents: str,
    source: str,
    flags: list[str],
    expected: str,
) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(f"[eval]\n{contents}\n")
    result = eval_cli.invoke(source, "openai", *flags, "--dry-run")
    assert result["runtime"] == expected
    assert "remote" not in EvalConfig.load(path).model_dump()


def test_legacy_remote_config_uses_hosted_runtime(eval_cli: _EvalCli, tmp_path: Path) -> None:
    (tmp_path / ".hud_eval.toml").write_text("[eval]\nremote = true\n")
    eval_cli.invoke("tasks.py", "openai", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HostedRuntime)


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


def test_load_treats_unset_settings_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / ".hud_eval.toml"
    path.write_text('[openai_compatible]\napi_key = "${openai_api_key}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$\{openai_api_key\} is not set"):
        EvalConfig.load(path)


@pytest.mark.parametrize(
    "contents, match",
    [
        ('[eval]\nmodle = "x"\n', "modle"),
        ('[eval]\nremote = "false"\n', "remote must be a boolean"),
        ('[eval]\nagent = "not-an-agent"\n', "claude"),
        ('[eval]\nruntime = "cloud"\n', "'local', 'hud', 'hosted', 'docker', 'modal' or 'daytona'"),
        ("[other]\nx = 1\n", "unknown sections: other"),
    ],
)
def test_load_rejects_invalid_configuration(tmp_path: Path, contents: str, match: str) -> None:
    path = tmp_path / ".hud_eval.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        EvalConfig.load(path)


def test_placement_defaults_from_source(eval_cli: _EvalCli) -> None:
    assert eval_cli.invoke("tasks.py", "openai", "--dry-run")["runtime"] == "local"
    assert eval_cli.invoke("My Tasks", "openai", "--dry-run")["runtime"] == "hosted"
    assert (
        eval_cli.invoke("My Tasks", "openai", "--runtime", "hud", "--dry-run")["runtime"] == "hud"
    )


def test_local_against_a_platform_taskset_is_refused(eval_cli: _EvalCli, monkeypatch) -> None:
    monkeypatch.setattr(
        Taskset,
        "from_api",
        classmethod(lambda cls, name: Taskset(name, [Task(env="demo", id="a")])),
    )
    payload = eval_cli.invoke("My Tasks", "openai", "--runtime", "local", "--yes", exit_code=2)
    assert "My Tasks is a platform taskset" in payload["message"]
    assert "--remote" in payload["message"]
    assert eval_cli.taskset is None


@pytest.mark.parametrize("flags", [["--runtime", "hud"], ["--remote"], ["--gateway"]])
def test_platform_features_require_hud_key(eval_cli: _EvalCli, monkeypatch, flags) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    payload = eval_cli.invoke("tasks.py", "gemini", *flags, "--yes", exit_code=1)
    assert payload["error"] == "permission_denied"
    assert eval_cli.taskset is None


def test_openai_compatible_requires_a_model(eval_cli: _EvalCli) -> None:
    payload = eval_cli.invoke("tasks.py", "openai_compatible", "--yes", exit_code=2)
    assert "Model name is required" in payload["message"]


def test_cli_model_wins_over_config_section(eval_cli: _EvalCli, tmp_path: Path) -> None:
    """``--model`` and the TOML section are passed to the agent verbatim; the positional
    agent argument is what resolves short names through the catalog."""
    (tmp_path / ".hud_eval.toml").write_text(
        '[openai_compatible]\nmodel = "moonshotai/kimi-k2.6"\n'
        "completion_kwargs = { temperature = 0.5 }\n",
        encoding="utf-8",
    )
    eval_cli.invoke("tasks.py", "openai_compatible", "--yes")
    assert eval_cli.agent.config.model == "moonshotai/kimi-k2.6"
    assert eval_cli.agent.config.completion_kwargs == {"temperature": 0.5}

    eval_cli.invoke(
        "tasks.py", "openai_compatible", "-m", "z-ai/glm-5.2", "--max-steps", "7", "--yes"
    )
    assert eval_cli.agent.config.model == "z-ai/glm-5.2"
    assert eval_cli.agent.config.max_steps == 7
    assert eval_cli.agent.config.completion_kwargs == {"temperature": 0.5}


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
    monkeypatch.setattr("hud.utils.gateway.list_gateway_models", lambda *_: [model])
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
    subprocesses: dict[object, MagicMock] = {}

    def subprocess_runtime(source: object) -> MagicMock:
        return subprocesses.setdefault(source, MagicMock(name=f"subprocess({source})"))

    monkeypatch.setattr(eval_mod, "DockerRuntime", lambda: docker)
    monkeypatch.setattr(eval_mod, "SubprocessRuntime", subprocess_runtime)
    image = Task(env="image", id="run", runtime_config=RuntimeConfig(image="example:latest"))
    data_row = Task(env="demo", id="plain")
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

    assert placement(bound) is subprocesses[(tmp_path / "mixed.py").resolve()].return_value
    assert placement(image) is docker.return_value
    # A row without a bound env is served from the tasks file's directory.
    assert placement(data_row) is subprocesses[tmp_path.resolve()].return_value


def test_json_rows_are_served_from_their_directory(
    eval_cli: _EvalCli, tmp_path: Path, monkeypatch
) -> None:
    sources: list[object] = []
    monkeypatch.setattr(
        eval_mod, "SubprocessRuntime", lambda source: sources.append(source) or MagicMock()
    )
    (tmp_path / "rows.json").write_text(
        '[{"env": "demo", "id": "a"}, {"env": "demo", "id": "b"}]', encoding="utf-8"
    )
    eval_cli.invoke("rows.json", "openai", "--all", "--yes")
    assert eval_cli.taskset is not None and len(eval_cli.taskset) == 2
    assert sources == [tmp_path.resolve()]


def test_explicit_placements(eval_cli: _EvalCli) -> None:
    eval_cli.invoke("tasks.py", "openai", "--runtime", "hud", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HUDRuntime)
    eval_cli.invoke("tasks.py", "openai", "--remote", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HostedRuntime)
    eval_cli.invoke("tasks.py", "openai", "--runtime", "hosted", "--yes")
    assert isinstance(eval_cli.kwargs["runtime"], HostedRuntime)
    eval_cli.invoke("tasks.py", "openai", "--runtime", "tcp://127.0.0.1:7000", "--yes")
    assert eval_cli.kwargs["runtime"] == Runtime("tcp://127.0.0.1:7000")
    for name, cls in (
        ("docker", DockerRuntime),
        ("modal", ModalRuntime),
        ("daytona", DaytonaRuntime),
    ):
        eval_cli.invoke("tasks.py", "openai", "--runtime", name, "--yes")
        assert isinstance(eval_cli.kwargs["runtime"], cls)


def test_unknown_runtime_lists_the_choices(eval_cli: _EvalCli) -> None:
    payload = eval_cli.invoke("tasks.py", "openai", "--runtime", "cloud", "--yes", exit_code=2)
    assert "'local', 'hud', 'hosted', 'docker', 'modal' or 'daytona'" in payload["message"]
    assert "URL scheme should be 'tcp'" not in payload["message"]  # 'cloud' is not url-shaped
    payload = eval_cli.invoke("tasks.py", "openai", "--runtime", "http://x:1", "--yes", exit_code=2)
    assert "URL scheme should be 'tcp'" in payload["message"]


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


@pytest.mark.parametrize("force_gateway", [False, True])
def test_openai_compatible_custom_endpoint_is_used_unless_gateway_is_forced(
    eval_cli: _EvalCli, monkeypatch, force_gateway
) -> None:
    client = MagicMock(return_value=object())
    gateway = MagicMock(return_value=object())
    monkeypatch.setattr("hud.agents.openai_compatible.agent.AsyncOpenAI", client)
    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", gateway)
    eval_cli.invoke(
        "tasks.py",
        "openai_compatible",
        "-m",
        "custom",
        "-c",
        "api_key=custom-key",
        "-c",
        "base_url=https://custom.example",
        *(["--gateway"] if force_gateway else []),
        "--yes",
    )
    if force_gateway:
        gateway.assert_called_once_with("openai")
        client.assert_not_called()
        assert eval_cli.agent.oai is gateway.return_value
    else:
        client.assert_called_once_with(api_key="custom-key", base_url="https://custom.example")
        gateway.assert_not_called()
        assert eval_cli.agent.oai is client.return_value


def test_bedrock_arn_in_config_selects_bedrock_client(eval_cli: _EvalCli, monkeypatch) -> None:
    monkeypatch.setattr(settings, "aws_access_key_id", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_access_key", "secret")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    bedrock = MagicMock(return_value=object())
    monkeypatch.setattr("anthropic.AsyncAnthropicBedrock", bedrock)
    eval_cli.invoke("tasks.py", "claude", "--config", f"checkpoint_name={_BEDROCK_ARN}", "--yes")
    assert eval_cli.agent.config.model == _BEDROCK_ARN
    assert eval_cli.agent.anthropic_client is bedrock.return_value


def test_interactive_picker_offers_current_catalog_models_newest_first(
    eval_cli: _EvalCli, monkeypatch
) -> None:
    catalog = _catalog(
        ("openai_compatible", "MiniMax M3", "MiniMax-M3", "2026-06-20T00:00:00Z"),
        ("openai_compatible", "Kimi K2.7", "moonshotai/kimi-k2.7", "2026-09-01T00:00:00Z"),
        ("claude", "Claude Opus 5", "claude-opus-5", "2026-07-27T00:00:00Z"),
    )
    catalog.append(
        GatewayModelInfo(
            id="old",
            name="Old",
            model_name="old",
            sdk_agent_type="openai_compatible",
            created_at="2026-09-10T00:00:00Z",
            deprecated_at="2026-09-11T00:00:00Z",
        )
    )
    monkeypatch.setattr("hud.cli.eval.list_gateway_models", lambda: catalog)
    offered: list[str] = []

    def pick(choices: list[Any]) -> Any:
        offered.extend(choice["name"] for choice in choices)
        return choices[0]["value"]

    monkeypatch.setattr(HUDConsole, "select", _picker("openai_compatible", pick))
    eval_cli.invoke("tasks.py", "--yes")

    assert offered == ["Kimi K2.7 (moonshotai/kimi-k2.7)", "MiniMax M3 (MiniMax-M3)"]
    assert eval_cli.agent.config.model == "moonshotai/kimi-k2.7"
    assert eval_cli.agent.config.model_name == "Kimi K2.7"


def test_interactive_picker_without_a_hud_key_uses_the_agent_default_model(
    eval_cli: _EvalCli, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", "provider-key")
    monkeypatch.setattr(
        "hud.cli.eval.list_gateway_models", lambda: pytest.fail("catalog needs a key")
    )
    monkeypatch.setattr(
        HUDConsole, "select", _picker("claude", lambda choices: pytest.fail("model prompt"))
    )
    eval_cli.invoke("tasks.py", "--yes")
    assert isinstance(eval_cli.agent.config, ClaudeConfig)
    assert eval_cli.agent.config.model == ClaudeConfig().model


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
    monkeypatch.setattr(
        "hud.cli.eval.list_gateway_models",
        lambda: _catalog(("openai", "GPT 5.6", "gpt-5.6", "2026-06-12T00:00:00Z")),
    )
    monkeypatch.setattr(
        HUDConsole, "select", _picker("openai", lambda choices: choices[0]["value"])
    )
    payload = eval_cli.invoke("--yes")
    assert payload["source"] == "rows.json"
    assert eval_cli.agent.config.model == "gpt-5.6"


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


@pytest.fixture
def local_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Callable[..., dict[str, Any]]]:
    async def answer(self: OpenAIAgent, run: Run) -> None:
        run.trace.content = "ok"

    monkeypatch.setattr(OpenAIAgent, "__call__", answer)
    monkeypatch.setattr(settings, "openai_api_key", "test-provider-key")
    monkeypatch.setenv("HUD_API_KEY", "")
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    original = dict(sys.modules)

    def invoke(source: Path | str, *flags: str) -> dict[str, Any]:
        result = CliRunner().invoke(
            app, ["eval", str(source), "openai", "--all", "--yes", "--json", *flags]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["error_count"] == 0, result.output
        assert payload["mean_reward"] == 1.0, result.output
        return payload

    yield invoke
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).is_relative_to(tmp_path):
            if name in original:
                sys.modules[name] = original[name]
            else:
                sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "layout",
    [
        "single",
        "standalone",
        "split",
        "hooks",
        "hooks_source",
        "assembled",
        "assembled_source",
        "json",
        "jsonl",
        "data_python",
        "directory",
        "assembled_directory",
        "package",
        "lazy_package",
    ],
)
def test_local_eval_project_layouts(
    local_eval: Callable[..., dict[str, Any]], tmp_path: Path, layout: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    events = project / "events"
    events.mkdir()
    (project / "asset.txt").write_text("ok")
    package = layout in {"package", "lazy_package"}
    prefix = "." if package else ""
    if package:
        (project / "__init__.py").write_text("")
    template = (
        "from pathlib import Path\n"
        '@env.template(id="solve")\n'
        "async def solve():\n"
        '    answer = yield "answer ok"\n'
        '    yield 1.0 if answer == Path("asset.txt").read_text() else 0.0\n'
    )
    hooks = (
        "import os\nfrom pathlib import Path\n"
        "@env.initialize\nasync def start():\n"
        '    Path(f"events/{os.getpid()}").write_text("started")\n'
        "@env.shutdown\nasync def stop():\n"
        '    Path(f"events/{os.getpid()}").write_text("stopped")\n'
    )
    if layout == "lazy_package":
        (project / "expected.py").write_text(
            'from pathlib import Path\nexpected = Path("asset.txt").read_text()\n'
        )
        template = template.replace(
            '    answer = yield "answer ok"\n',
            '    from .expected import expected\n    answer = yield "answer ok"\n',
        ).replace('Path("asset.txt").read_text()', "expected")
    if layout in {
        "hooks",
        "hooks_source",
        "assembled",
        "assembled_source",
        "assembled_directory",
        "package",
    }:
        (project / "local_core.py").write_text(
            'from hud import Environment\nenv = Environment("local-test")\n'
        )
        (project / "local_templates.py").write_text(
            f"from {prefix}local_core import env\n" + template
        )
        (project / "local_extra.py").write_text(
            f"from {prefix}local_core import env\n"
            '@env.template(id="extra")\nasync def extra():\n    yield "extra"\n    yield 1.0\n'
        )
        env_source = (
            f"from {prefix}local_core import env\n"
            f"from {prefix}local_templates import solve\n"
            + (f"from {prefix}local_extra import extra\n" if not layout.startswith("hooks") else "")
            + hooks
        )
    else:
        env_source = (
            'from hud import Environment\nenv = Environment("local-test")\n' + template + hooks
        )
    if layout in {"single", "standalone", "lazy_package", "hooks_source", "assembled_source"}:
        env_source += "tasks = [solve()]\n"
        source = project / "env.py"
    else:
        (project / "tasks.py").write_text(
            f"from {prefix}env import env, solve\ntasks = [solve()]\n"
        )
        source = project / "tasks.py"
    (project / "env.py").write_text(env_source)
    if layout == "standalone":
        source = project / "standalone.py"
        source.write_text(env_source)
        (project / "env.py").write_text(
            'from hud import Environment\nenv = Environment("unrelated")\n'
        )
    if layout in {"json", "jsonl"}:
        source = project / f"tasks.{layout}"
        row = {"env": "local-test", "id": "solve"}
        source.write_text(json.dumps([row] if layout == "json" else row))
    if layout == "data_python":
        source.write_text(
            'from hud.eval import Task\ntasks = [Task(env="local-test", id="solve")]\n'
        )
    if layout in {"directory", "assembled_directory"}:
        source = project
    payload = local_eval(source, "--group", "2", "--max-concurrent", "2")
    assert payload["run_count"] == 2
    assert len(list(events.iterdir())) == 2
    assert {event.read_text() for event in events.iterdir()} == {"stopped"}


@pytest.mark.parametrize(
    "source,same_name",
    [
        ("tasks.py", False),
        ("tasks.py", True),
        (".", False),
        (".", True),
        ("rows.json", False),
        ("rows.json", True),
        ("rows.jsonl", False),
    ],
)
def test_local_eval_environment_selection(
    local_eval: Callable[..., dict[str, Any]], tmp_path: Path, source: str, same_name: bool
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "bound_env.py").write_text(
        'from hud import Environment\nenv = Environment("bound")\n'
        '@env.template(id="solve")\nasync def solve():\n'
        '    answer = yield "answer ok"\n    yield float(answer == "ok")\n'
    )
    name = "bound" if same_name else "unrelated"
    (project / "env.py").write_text(
        f"from hud import Environment\nenv = Environment({name!r})\n"
        '@env.template(id="solve")\nasync def solve():\n'
        '    yield "unrelated"\n    yield 0.0\n'
    )
    (project / "tasks.py").write_text("from bound_env import solve\ntasks = [solve()]\n")
    if source in {"rows.json", "rows.jsonl"}:
        (project / source).write_text('{"env": "bound", "id": "solve"}\n')
    if same_name and source != "tasks.py":
        result = CliRunner().invoke(
            app, ["eval", str(project / source), "openai", "--all", "--yes", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["error_count"] == 1
        assert "multiple Environments" in result.output
    else:
        assert local_eval(project / source)["run_count"] == 1


@pytest.mark.parametrize("shared", [True, False])
def test_local_eval_places_nested_verifier(
    local_eval: Callable[..., dict[str, Any]], tmp_path: Path, shared: bool
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "local_core.py").write_text(
        'from hud import Environment\nactor = Environment("actor")\n'
        + ("judge = actor\n" if shared else 'judge = Environment("judge")\n')
    )
    (project / "local_actor.py").write_text(
        "from local_core import actor\n"
        '@actor.template(id="solve")\nasync def solve():\n'
        '    answer = yield "answer ok"\n    yield {"score": 0.0, "answer": answer}\n'
    )
    (project / "local_judge.py").write_text(
        "from local_core import judge\n"
        '@judge.template(id="verify")\nasync def verify():\n'
        '    result = yield ""\n    yield 1.0 if result["answer"] == "ok" else 0.0\n'
    )
    (project / "env.py").write_text(
        "from local_core import actor, judge\n"
        "from local_actor import solve\nfrom local_judge import verify\n"
    )
    (project / "tasks.py").write_text(
        "from env import solve, verify\ntasks = [solve()]\ntasks[0].verifier = verify()\n"
    )
    assert local_eval(project / "tasks.py")["run_count"] == 1


@pytest.mark.parametrize("container", ["image", "compose"])
def test_platform_container_rows_can_use_local_placement(
    eval_cli: _EvalCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("services:\n  main:\n    image: example:latest\n")
    config = (
        RuntimeConfig(image="example:latest")
        if container == "image"
        else RuntimeConfig.model_validate(
            {"compose": {"document": str(compose), "root": str(tmp_path)}}
        )
    )
    task = Task(env="demo", id="solve", runtime_config=config)
    monkeypatch.setattr(Taskset, "from_api", classmethod(lambda cls, name: Taskset(name, [task])))
    docker = MagicMock()
    monkeypatch.setattr(eval_mod, "DockerRuntime", lambda: docker)
    eval_cli.invoke("Platform Tasks", "openai", "--runtime", "local", "--yes")
    assert eval_cli.kwargs["runtime"](task) is docker.return_value
    docker.assert_called_once_with(task)


@pytest.mark.parametrize("shared,configured", [(True, False), (False, True), (False, False)])
def test_platform_local_placement_checks_separate_verifier(
    eval_cli: _EvalCli, monkeypatch: pytest.MonkeyPatch, shared: bool, configured: bool
) -> None:
    config = RuntimeConfig(image="example:latest")
    verifier = Task(
        env="actor" if shared else "judge",
        id="verify",
        runtime_config=config if configured else None,
    )
    task = Task(env="actor", id="solve", runtime_config=config, verifier=verifier)
    monkeypatch.setattr(Taskset, "from_api", classmethod(lambda cls, name: Taskset(name, [task])))
    payload = eval_cli.invoke(
        "Platform Tasks",
        "openai",
        "--runtime",
        "local",
        "--yes",
        exit_code=0 if shared or configured else 2,
    )
    if not shared and not configured:
        assert "no env source to spawn locally" in payload["message"]
        assert eval_cli.taskset is None
