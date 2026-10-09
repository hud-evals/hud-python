"""``hud eval``: plan, validate and run real rollouts of a taskset with a scripted model."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsFloat, IsInstance, IsInt, IsStr
from inline_snapshot import snapshot

from hud import Environment
from hud.eval import LocalRuntime, Task
from tests.harness import say

from .conftest import API_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, HudEnv, Models

HEX_ID = IsStr(regex=r"[0-9a-f]{32}")
RUNTIMES = r"Input should be 'local', 'hud', 'hosted', 'docker', 'modal' or 'daytona'"
TCP_ONLY = r"URL scheme should be 'tcp'"
TASKS_PY = """\
from hud import Environment

env = Environment("demo")


@env.template(id="solve")
async def solve(n: int = 0):
    answer = yield f"solve {n}"
    yield 1.0 if answer == "ok" else 0.0


tasks = [solve(n=0), solve(n=1)]
"""


@pytest.fixture
def gateway(models: Models, hud_env: HudEnv) -> Models:
    """A signed-in user whose agents reach a scripted model that always answers "ok"."""
    hud_env.set(HUD_API_KEY=API_KEY)
    models.script([say("ok")])
    return models


@pytest.fixture
def tasks(hud: Hud) -> Path:
    path = hud.cwd / "tasks.py"
    path.write_text(TASKS_PY)
    return path


def run(hud: Hud, *argv: str, env: dict[str, str] | None = None) -> Any:
    """``hud eval ... --yes --json``; the result document of a successful run."""
    result = hud("eval", *argv, "--yes", "--json", env=env)
    assert result.exit_code == 0, result
    return result.json


# ─── results ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("grader", "exit_code", "document"),
    [
        pytest.param(
            'yield 1.0 if answer == "ok" else 0.0',
            0,
            {
                "mean_reward": 1.0,
                "error_count": 0,
                "runs": [("solve-5fe4d610", 1.0, False), ("solve-bae34777", 1.0, False)],
            },
            id="graded",
        ),
        pytest.param(
            'yield 1.0 if answer == "something else" else 0.0',
            0,
            {
                "mean_reward": 0.5,
                "error_count": 0,
                "runs": [("solve-5fe4d610", 1.0, False), ("solve-bae34777", 0.0, False)],
            },
            id="low-reward-is-not-an-error",
        ),
    ],
)
def test_eval_reports_every_run_and_fails_when_one_errored(
    hud: Hud, gateway: Models, grader: str, exit_code: int, document: dict[str, Any]
) -> None:
    (hud.cwd / "tasks.py").write_text(
        TASKS_PY.replace(
            '    answer = yield f"solve {n}"\n    yield 1.0 if answer == "ok" else 0.0\n',
            f'    answer = yield f"solve {{n}}"\n    if n == 0:\n        yield 1.0\n    {grader}\n',
        )
    )

    result = hud(
        "eval", "tasks.py", "openai_compatible", "-m", "scripted", "--all", "--yes", "--json"
    )

    assert result.exit_code == exit_code, result
    assert result.json == {
        "job_id": HEX_ID,
        "source": "tasks.py",
        "elapsed_seconds": IsFloat(gt=0),
        "run_count": 2,
        "mean_reward": IsFloat(),
        "error_count": IsInt(),
        "runs": [
            {
                "task_id": "solve",
                "slug": IsStr(),
                "reward": IsFloat(),
                "is_error": IsInstance(bool),
                "trace_id": HEX_ID,
            }
        ]
        * 2,
    }
    assert {
        "mean_reward": result.json["mean_reward"],
        "error_count": result.json["error_count"],
        "runs": [(run["slug"], run["reward"], run["is_error"]) for run in result.json["runs"]],
    } == document


@pytest.mark.parametrize(
    ("argv", "prompts", "warning"),
    [
        ([], ["solve 0"], "Running only 1 of 2 tasks (the first)."),
        (["--all"], ["solve 0", "solve 1"], None),
        (["--task-ids", "1"], ["solve 1"], None),
        (["--task-ids", "solve"], ["solve 0", "solve 1"], None),
        (["--group", "3", "--max-concurrent", "2"], ["solve 0"] * 3, None),
    ],
)
def test_eval_selects_the_tasks_to_run(
    hud: Hud,
    gateway: Models,
    tasks: Path,
    argv: list[str],
    prompts: list[str],
    warning: str | None,
) -> None:
    result = hud(
        "eval", "tasks.py", "openai_compatible", "-m", "scripted", *argv, "--yes", "--json"
    )

    assert result.exit_code == 0, result
    assert result.json["run_count"] == len(prompts)
    assert result.json["mean_reward"] == 1.0
    assert sorted(request.prompt for request in gateway.requests()) == prompts
    assert (warning is not None and warning in " ".join(result.stderr.split())) or warning is None


# ─── how the agent reaches its model ────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "config", "environment", "expected"),
    [
        pytest.param(
            ["openai_compatible", "-m", "scripted"],
            "",
            {},
            {"protocol": "chat", "model": "scripted", "key": API_KEY},
            id="gateway-by-default",
        ),
        pytest.param(
            ["openai_compatible"],
            '[openai_compatible]\nmodel = "moonshotai/kimi-k2.6"\n'
            "completion_kwargs = { temperature = 0.5 }\n",
            {},
            {"protocol": "chat", "model": "moonshotai/kimi-k2.6", "temperature": 0.5},
            id="toml-section",
        ),
        pytest.param(
            ["openai_compatible", "-m", "z-ai/glm-5.2"],
            '[openai_compatible]\nmodel = "moonshotai/kimi-k2.6"\n'
            "completion_kwargs = { temperature = 0.5 }\n",
            {},
            {"protocol": "chat", "model": "z-ai/glm-5.2", "temperature": 0.5},
            id="model-flag-wins-over-toml",
        ),
        pytest.param(
            ["glm-5.2"],
            "",
            {},
            {"protocol": "chat", "model": "z-ai/glm-5.2", "key": API_KEY},
            id="catalog-alias",
        ),
        pytest.param(
            ["claude", "--config", "max_tokens=100"],
            "",
            {},
            {"protocol": "anthropic", "max_tokens": 100, "key": API_KEY},
            id="config-flag",
        ),
        pytest.param(
            ["claude", "-c", "claude.max_tokens=200"],
            "",
            {},
            {"protocol": "anthropic", "max_tokens": 200},
            id="config-flag-for-a-named-agent",
        ),
        pytest.param(
            ["openai"],
            "",
            {"OPENAI_API_KEY": "provider-key"},
            {"protocol": "responses", "key": "provider-key"},
            id="provider-key-wins",
        ),
        pytest.param(
            ["openai", "--gateway"],
            "",
            {"OPENAI_API_KEY": "provider-key"},
            {"protocol": "responses", "key": API_KEY},
            id="gateway-flag-wins-over-provider-key",
        ),
        pytest.param(
            ["claude"],
            "",
            {"ANTHROPIC_API_KEY": "provider-key"},
            {"protocol": "anthropic", "key": "provider-key"},
            id="anthropic-provider-key",
        ),
        pytest.param(
            ["openai_compatible", "--model", "MiniMax-M3"],
            "",
            {"OPENAI_API_KEY": "provider-key"},
            {"protocol": "chat", "model": "MiniMax-M3", "key": API_KEY},
            id="compatible-models-ignore-the-openai-key",
        ),
        pytest.param(
            ["openai_compatible", "-m", "custom", "-c", "api_key=custom-key", "-c", "base_url=URL"],
            "",
            {},
            {"protocol": "chat", "model": "custom", "key": "custom-key"},
            id="custom-endpoint",
        ),
        pytest.param(
            [
                *["openai_compatible", "-m", "custom", "-c", "api_key=custom-key"],
                *["-c", "base_url=URL", "--gateway"],
            ],
            "",
            {},
            {"protocol": "chat", "model": "custom", "key": API_KEY},
            id="gateway-flag-wins-over-custom-endpoint",
        ),
        pytest.param(
            ["gemini"],
            "",
            {},
            {"protocol": "gemini", "key": API_KEY},
            id="gemini-through-the-gateway",
        ),
    ],
)
def test_the_agent_reaches_its_model_with_the_right_key_and_settings(
    hud: Hud,
    services: FakeServices,
    gateway: Models,
    hud_env: HudEnv,
    tasks: Path,
    argv: list[str],
    config: str,
    environment: dict[str, str],
    expected: dict[str, Any],
) -> None:
    """Provider SDKs read their own ``*_BASE_URL``, which here points at the scripted model."""
    gateway_url = gateway.url
    services.route("api", "GET", "/v2/models", json={"items": [GLM], "total": 1})
    (hud.cwd / ".hud_eval.toml").write_text(config)
    hud_env.set(OPENAI_BASE_URL=gateway_url, ANTHROPIC_BASE_URL=gateway_url, **environment)

    # The same scripted model under another host name, so it is not the HUD gateway.
    custom = gateway_url.replace("127.0.0.1", "localhost")
    document = run(hud, "tasks.py", *[arg.replace("URL", custom) for arg in argv])

    assert document["mean_reward"] == 1.0
    (request,) = gateway.requests()
    observed = {
        "protocol": request.protocol,
        "model": request.model,
        "key": key(request.headers),
        "temperature": request.body.get("temperature"),
        "max_tokens": request.body.get("max_tokens"),
    }
    assert {name: observed[name] for name in expected} == expected


GLM = {
    "id": "z-ai/glm-5.2",
    "model_name": "z-ai/glm-5.2",
    "sdk_agent_type": "openai_compatible",
    "provider": {"name": "openai"},
}


def key(headers: dict[str, str]) -> str | None:
    """The credential a provider SDK sent, whichever header its protocol uses."""
    for name in ("x-api-key", "x-goog-api-key"):
        if name in headers:
            return headers[name]
    return headers.get("authorization", "").removeprefix("Bearer ") or None


# ─── project layouts run locally ────────────────────────────────────────

TEMPLATE = (
    "from pathlib import Path\n"
    '@env.template(id="solve")\n'
    "async def solve():\n"
    '    answer = yield "answer ok"\n'
    '    yield 1.0 if answer == Path("asset.txt").read_text() else 0.0\n'
)
HOOKS = (
    "import os\nfrom pathlib import Path\n"
    "@env.initialize\nasync def start():\n"
    '    Path(f"events/{os.getpid()}").write_text("started")\n'
    "@env.shutdown\nasync def stop():\n"
    '    Path(f"events/{os.getpid()}").write_text("stopped")\n'
)


def project_layout(project: Path, layout: str) -> Path:
    """Write one way of laying out an env and its tasks; return the source to evaluate."""
    project.mkdir()
    (project / "events").mkdir()
    (project / "asset.txt").write_text("ok")
    package = layout in {"package", "lazy_package"}
    prefix = "." if package else ""
    if package:
        (project / "__init__.py").write_text("")
    template = TEMPLATE
    if layout == "lazy_package":
        (project / "expected.py").write_text(
            'from pathlib import Path\nexpected = Path("asset.txt").read_text()\n'
        )
        template = template.replace(
            '    answer = yield "answer ok"\n',
            '    from .expected import expected\n    answer = yield "answer ok"\n',
        ).replace('Path("asset.txt").read_text()', "expected")
    split = {"hooks", "hooks_source", "assembled", "assembled_source", "assembled_directory"}
    if layout in split | {"package"}:
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
            + HOOKS
        )
    else:
        env_source = (
            'from hud import Environment\nenv = Environment("local-test")\n' + template + HOOKS
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
    return source


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
def test_each_project_layout_runs_in_fresh_env_processes(
    hud: Hud, gateway: Models, tmp_path: Path, layout: str
) -> None:
    source = project_layout(tmp_path / "project", layout)

    document = run(
        hud,
        str(source),
        "openai_compatible",
        "-m",
        "scripted",
        "--all",
        "--group",
        "2",
        "--max-concurrent",
        "2",
    )

    assert (document["run_count"], document["mean_reward"], document["error_count"]) == (2, 1.0, 0)
    events = list((tmp_path / "project" / "events").iterdir())
    assert len(events) == 2
    assert {event.read_text() for event in events} == {"stopped"}


@pytest.mark.parametrize(
    ("source", "same_name", "errors"),
    [
        ("tasks.py", False, 0),
        ("tasks.py", True, 0),
        (".", False, 0),
        (".", True, 1),
        ("rows.json", False, 0),
        ("rows.json", True, 1),
        ("rows.jsonl", False, 0),
    ],
)
def test_a_row_runs_in_the_environment_it_names(
    hud: Hud, gateway: Models, tmp_path: Path, source: str, same_name: bool, errors: int
) -> None:
    """Rows resolve their env by name; two envs sharing that name make the row an error."""
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
    if source.startswith("rows"):
        (project / source).write_text('{"env": "bound", "id": "solve"}\n')

    result = hud(
        "eval",
        str(project / source),
        "openai_compatible",
        "-m",
        "scripted",
        "--all",
        "--yes",
        "--json",
    )

    assert result.json["run_count"] == 1
    assert result.json["error_count"] == errors
    assert result.exit_code == errors, result
    assert ("multiple Environments" in result.stderr) is bool(errors)


@pytest.mark.parametrize("shared", [True, False])
def test_a_nested_verifier_runs_in_its_own_environment(
    hud: Hud, gateway: Models, tmp_path: Path, shared: bool
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

    document = run(hud, str(project / "tasks.py"), "openai_compatible", "-m", "scripted")

    assert (document["run_count"], document["mean_reward"]) == (1, 1.0)


async def test_eval_attaches_to_an_already_served_environment(
    hud: Hud, gateway: Models, tmp_path: Path
) -> None:
    env = Environment("served")

    @env.template(id="solve")
    async def solve():
        answer = yield "answer ok"
        yield 1.0 if answer == "ok" else 0.0

    (hud.cwd / "rows.json").write_text('[{"env": "served", "id": "solve"}]')
    async with LocalRuntime(env)(Task(env="served", id="solve")) as runtime:
        document = await asyncio.to_thread(
            run, hud, "rows.json", "openai_compatible", "-m", "scripted", "--runtime", runtime.url
        )

    assert (document["run_count"], document["mean_reward"]) == (1, 1.0)


# ─── planning and validation ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "argv", "plan"),
    [
        (
            "",
            ["tasks.py", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "local",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "",
            ["My Tasks", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "My Tasks",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hosted",
                    "remote": True,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "",
            ["My Tasks", "openai", "--runtime", "hud"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "My Tasks",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hud",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "",
            ["tasks.py", "openai", "--full"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "local",
                    "remote": False,
                    "all": True,
                    "max_steps": 100,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "[eval]\nremote = true\n",
            ["tasks.py", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hosted",
                    "remote": True,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "[eval]\nremote = false\n",
            ["My Tasks", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "My Tasks",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hosted",
                    "remote": True,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            '[eval]\nremote = true\nruntime = "local"\n',
            ["tasks.py", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "local",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            '[eval]\nremote = false\nruntime = "hud"\n',
            ["tasks.py", "openai"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hud",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "[eval]\nremote = true\n",
            ["tasks.py", "openai", "--runtime", "local"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "local",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            "[eval]\nremote = false\n",
            ["tasks.py", "openai", "--remote"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hosted",
                    "remote": True,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
        (
            '[eval]\nsource = "tasks.py"\nagent = "openai"\nmax_steps = 5\nruntime = "hosted"\n'
            'group_size = 2\ntask_ids = ["solve"]\n',
            [],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": None,
                    "runtime": "hosted",
                    "remote": True,
                    "all": False,
                    "max_steps": 5,
                    "max_concurrent": 30,
                    "group_size": 2,
                    "task_ids": ["solve"],
                }
            ),
        ),
        (
            '[openai]\nmodel = "${MY_EVAL_MODEL}"\n',
            ["tasks.py", "openai", "-m", "x"],
            snapshot(
                {
                    "dry_run": True,
                    "action": "eval",
                    "source": "tasks.py",
                    "agent": "openai",
                    "model": "x",
                    "runtime": "local",
                    "remote": False,
                    "all": False,
                    "max_steps": 10,
                    "max_concurrent": 30,
                    "group_size": 1,
                    "task_ids": None,
                }
            ),
        ),
    ],
)
def test_a_dry_run_prints_the_plan_and_runs_nothing(
    hud: Hud,
    services: FakeServices,
    hud_env: HudEnv,
    tasks: Path,
    config: str,
    argv: list[str],
    plan: dict[str, Any],
) -> None:
    hud_env.set(MY_EVAL_MODEL="gpt-4o")
    (hud.cwd / ".hud_eval.toml").write_text(config)

    result = hud("eval", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json == plan
    assert services.requests() == []
    assert (hud.cwd / ".hud_eval.toml").read_text() == config


@pytest.mark.parametrize(
    ("config", "argv", "key", "exit_code", "document"),
    [
        (
            '[eval]\nmodle = "x"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": IsStr(regex=r"(?s).*\nmodle\n  Extra inputs are not permitted .*"),
                }
            ),
        ),
        (
            '[eval]\nremote = "false"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": ".hud_eval.toml: remote must be a boolean"}),
        ),
        (
            '[eval]\nagent = "not-an-agent"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": IsStr(
                        regex=r"(?s).*\nagent\n  Input should be 'claude', 'claude_cli', .*"
                    ),
                }
            ),
        ),
        (
            '[eval]\nruntime = "cloud"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": IsStr(regex=f"(?s)(?!.*{TCP_ONLY}).*{RUNTIMES}.*"),
                }
            ),
        ),
        (
            "[other]\nx = 1\n",
            ["--dry-run"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": ".hud_eval.toml: unknown sections: other"}),
        ),
        (
            "[eval",
            ["--dry-run"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Expected ']' at the end of a table declaration (at end of document)"
                    ),
                }
            ),
        ),
        (
            '[openai]\nmodel = "${HUD_EVAL_TEST_UNSET}"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot(
                {"error": "usage", "message": ".hud_eval.toml: ${HUD_EVAL_TEST_UNSET} is not set"}
            ),
        ),
        (
            '[openai_compatible]\napi_key = "${openai_api_key}"\n',
            ["--dry-run"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": ".hud_eval.toml: ${openai_api_key} is not set"}),
        ),
        (
            "",
            ["--runtime", "http://x:1", "--dry-run"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": IsStr(regex=f"(?s).*{TCP_ONLY}.*"),
                }
            ),
        ),
        (
            "",
            ["--runtime", "hud", "--remote"],
            API_KEY,
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "--runtime and --remote are mutually exclusive placement options",
                }
            ),
        ),
        (
            "",
            ["-c", "max_tokens"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": "--config expects key=value, got 'max_tokens'"}),
        ),
        (
            "",
            ["-c", "robot.max_tokens=1"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": "'robot' is not a valid AgentType"}),
        ),
        (
            "",
            ["--runtime", "hud"],
            None,
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "HUD_API_KEY is required",
                    "suggestion": "Run 'hud set HUD_API_KEY=your-key-here'.",
                }
            ),
        ),
        (
            "",
            ["--remote"],
            None,
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "HUD_API_KEY is required",
                    "suggestion": "Run 'hud set HUD_API_KEY=your-key-here'.",
                }
            ),
        ),
        (
            "",
            ["--gateway"],
            None,
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "HUD_API_KEY is required",
                    "suggestion": "Run 'hud set HUD_API_KEY=your-key-here'.",
                }
            ),
        ),
        (
            "",
            ["--task-ids", "nope"],
            API_KEY,
            2,
            snapshot({"error": "usage", "message": "No tasks matching: nope"}),
        ),
    ],
)
def test_invalid_configuration_is_an_error_document(
    hud: Hud,
    services: FakeServices,
    hud_env: HudEnv,
    tasks: Path,
    config: str,
    argv: list[str],
    key: str | None,
    exit_code: int,
    document: dict[str, Any],
) -> None:
    hud_env.set(HUD_API_KEY=key)
    (hud.cwd / ".hud_eval.toml").write_text(config)

    result = hud("eval", "tasks.py", "openai", *argv, "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert services.requests() == []
    assert (hud.cwd / ".hud_eval.toml").read_text() == config


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["--dry-run"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Dry-run requires an explicit task source and agent (or configured "
                        "defaults)."
                    ),
                }
            ),
        ),
        (
            ["tasks.py", "--config", "max_tokens=1", "--yes"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "--config max_tokens=... needs an agent; pass one or write "
                        "<agent>.max_tokens=..."
                    ),
                }
            ),
        ),
        (
            ["tasks.py", "openai_compatible", "--yes"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Model name is required for OpenAI compatible agent; use --model.",
                }
            ),
        ),
        (
            ["--yes"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No task JSON or JSONL files found in current directory",
                }
            ),
        ),
    ],
)
def test_missing_inputs_are_errors_without_prompting(
    hud: Hud, services: FakeServices, argv: list[str], exit_code: int, document: dict[str, Any]
) -> None:
    if "tasks.py" in argv:
        (hud.cwd / "tasks.py").write_text(TASKS_PY)

    result = hud("eval", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document


def test_a_python_source_without_tasks_is_imported_on_the_main_thread(
    hud: Hud, gateway: Models
) -> None:
    (hud.cwd / "probe.py").write_text(
        "import threading\nfrom pathlib import Path\n"
        'Path("main-thread.txt").write_text('
        "str(threading.current_thread() is threading.main_thread()))\n"
        "tasks = []\n"
    )

    result = hud("eval", "probe.py", "openai", "--yes", "--json")

    assert result.exit_code == 2, result
    assert "No runnable Tasks" in result.json["message"]
    assert (hud.cwd / "main-thread.txt").read_text() == "True"


def test_one_tasks_file_is_picked_when_no_source_is_given(hud: Hud, gateway: Models) -> None:
    (hud.cwd / "env.py").write_text(TASKS_PY)
    (hud.cwd / "rows.json").write_text('[{"env": "demo", "id": "solve", "args": {"n": 4}}]')
    (hud.cwd / ".hidden.json").write_text("[]")
    (hud.cwd / ".hud_eval.toml").write_text('[eval]\nagent = "openai_compatible"\nmodel = "m"\n')

    document = run(hud)

    assert (document["source"], document["mean_reward"]) == ("rows.json", 1.0)
    assert [request.prompt for request in gateway.requests()] == ["solve 4"]


# ─── platform tasksets ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task",
    [
        pytest.param({"name": "one", "env": "demo", "scenario": "solve"}, id="no-container"),
        pytest.param(
            {
                "name": "one",
                "env": "actor",
                "scenario": "solve",
                "runtime_config": {"image": "example:latest"},
                "verifier": {"env": "judge", "id": "verify"},
            },
            id="verifier-without-container",
        ),
    ],
)
def test_a_platform_taskset_runs_locally_only_from_container_rows(
    hud: Hud, services: FakeServices, hud_env: HudEnv, task: dict[str, Any]
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    taskset = "eeeeeeee-0000-4000-8000-000000000001"
    services.route(
        "api", "GET", "/v2/tasksets/by-name/{name}", json={"taskset_id": taskset, "name": "Demo"}
    )
    services.route(
        "api", "GET", f"/v2/tasksets/{taskset}/export", json={"name": "Demo", "tasks": [task]}
    )

    result = hud("eval", "Demo", "openai", "--runtime", "local", "--yes", "--json")

    assert result.exit_code == 2, result
    assert result.json == {
        "error": "usage",
        "message": "Demo is a platform taskset, so there is no env source to spawn locally. "
        "Run it with --remote, --runtime hud, or --runtime tcp://host:port.",
    }
