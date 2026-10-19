"""``hud eval``: plan, validate and run real rollouts of a taskset with a scripted model."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.eval import LocalRuntime, Task
from tests.support import say, write_files
from tests.support.platform import API_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, HudEnv, Models, Platform

TASKS_PY = """\
from hud import Environment

env = Environment("demo")


@env.template(id="solve")
async def solve(n: int = 0):
    answer = yield f"solve {n}"
    yield 1.0 if answer == "ok" else 0.0


tasks = [solve(n=0), solve(n=1)]
"""
SCRIPTED = ("openai_compatible", "-m", "scripted")


@pytest.fixture
def gateway(models: Models, platform: Platform) -> Models:
    """A signed-in user whose agents reach a scripted model that always answers "ok"."""
    models.script([say("ok")])
    return models


@pytest.fixture
def tasks(hud: Hud) -> Path:
    path = hud.cwd / "tasks.py"
    path.write_text(TASKS_PY)
    return path


def run(hud: Hud, *argv: str) -> Any:
    """``hud eval ... --yes --json``; the result document of a successful run."""
    result = hud("eval", *argv, "--yes", "--json")
    assert result.exit_code == 0, result
    return result.json


# ─── results ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("grade", "exit_code", "rewards", "errors"),
    [
        pytest.param("0.0", 0, [1.0, 0.0], [False, False], id="a-low-reward-is-not-an-error"),
        pytest.param(
            '{"score": 0.0, "isError": True, "content": "grader unavailable"}',
            1,
            [1.0, 0.0],
            [False, True],
            id="a-grade-error",
        ),
        pytest.param("1 / 0", 1, [1.0, 0.0], [False, True], id="a-grader-crash"),
    ],
)
def test_eval_reports_every_run_and_fails_when_one_errored(
    hud: Hud,
    gateway: Models,
    grade: str,
    exit_code: int,
    rewards: list[float],
    errors: list[bool],
) -> None:
    (hud.cwd / "tasks.py").write_text(
        'from hud import Environment\nenv = Environment("demo")\n'
        '@env.template(id="solve")\nasync def solve(n: int = 0):\n'
        '    answer = yield f"solve {n}"\n'
        f"    yield 1.0 if n == 0 else {grade}\n"
        "tasks = [solve(n=0), solve(n=1)]\n"
    )

    result = hud("eval", "tasks.py", *SCRIPTED, "--all", "--yes", "--json")

    assert result.exit_code == exit_code, result
    runs = sorted(result.json["runs"], key=lambda run: run["is_error"])
    assert [run["reward"] for run in runs] == rewards
    assert [run["is_error"] for run in runs] == errors
    assert result.json["error_count"] == sum(errors)


@pytest.mark.parametrize(
    ("argv", "prompts"),
    [
        pytest.param([], ["solve 0"], id="the-first-by-default"),
        pytest.param(["--all"], ["solve 0", "solve 1"], id="all"),
        pytest.param(["--task-ids", "1"], ["solve 1"], id="by-index"),
    ],
)
def test_eval_runs_the_tasks_it_selects(
    hud: Hud, gateway: Models, tasks: Path, argv: list[str], prompts: list[str]
) -> None:
    run(hud, "tasks.py", *SCRIPTED, *argv)

    assert sorted(request.prompt for request in gateway.requests()) == prompts


# ─── how the agent reaches its model ────────────────────────────────────


GLM = {
    "id": "z-ai/glm-5.2",
    "model_name": "z-ai/glm-5.2",
    "sdk_agent_type": "openai_compatible",
    "provider": {"name": "openai"},
}
KIMI_TOML = (
    '[openai_compatible]\nmodel = "moonshotai/kimi-k2.6"\n'
    "completion_kwargs = { temperature = 0.5 }\n"
)


@pytest.mark.parametrize(
    ("argv", "config", "expected"),
    [
        pytest.param(
            list(SCRIPTED),
            "",
            {"key": API_KEY},
            id="gateway-by-default",
        ),
        pytest.param(
            ["openai_compatible"],
            KIMI_TOML,
            {"model": "moonshotai/kimi-k2.6", "temperature": 0.5},
            id="toml-section",
        ),
        pytest.param(
            ["openai_compatible", "-m", "z-ai/glm-5.2"],
            KIMI_TOML,
            {"model": "z-ai/glm-5.2", "temperature": 0.5},
            id="model-flag-wins-over-toml",
        ),
        pytest.param(
            ["glm-5.2"],
            "",
            {"protocol": "chat", "model": "z-ai/glm-5.2"},
            id="catalog-alias",
        ),
        pytest.param(
            ["claude", "--config", "max_tokens=100"],
            "",
            {"max_tokens": 100, "key": API_KEY},
            id="config-flag",
        ),
        pytest.param(
            ["openai_compatible", "-m", "custom", "-c", "api_key=custom-key", "-c", "base_url=URL"],
            "",
            {"key": "custom-key"},
            id="custom-endpoint",
        ),
    ],
)
def test_the_agent_reaches_its_model_with_the_right_key_and_settings(
    hud: Hud,
    platform: Platform,
    gateway: Models,
    hud_env: HudEnv,
    tasks: Path,
    argv: list[str],
    config: str,
    expected: dict[str, Any],
) -> None:
    """Provider SDKs read their own ``*_BASE_URL``, which here points at the scripted model."""
    platform.add_model(**GLM)
    (hud.cwd / ".hud_eval.toml").write_text(config)
    hud_env.set(OPENAI_BASE_URL=gateway.url, ANTHROPIC_BASE_URL=gateway.url)

    # The same scripted model under another host name, so it is not the HUD gateway.
    custom = gateway.url.replace("127.0.0.1", "localhost")
    run(hud, "tasks.py", *[arg.replace("URL", custom) for arg in argv])

    (request,) = gateway.requests()
    credential = (
        request.headers.get("x-api-key")
        or request.headers.get("x-goog-api-key")
        or request.headers.get("authorization", "").removeprefix("Bearer ")
    )
    observed = {
        "protocol": request.protocol,
        "model": request.model,
        "key": credential,
        "temperature": request.body.get("temperature"),
        "max_tokens": request.body.get("max_tokens"),
    }
    assert {name: observed[name] for name in expected} == expected


# ─── project layouts run locally ────────────────────────────────────────

IMPORTS = "import os\nfrom pathlib import Path\n"
ENV = 'from hud import Environment\nenv = Environment("local-test")\n'
SOLVE = (
    '@env.template(id="solve")\nasync def solve():\n'
    '    answer = yield "answer ok"\n'
    '    yield 1.0 if answer == Path("asset.txt").read_text() else 0.0\n'
)
HOOKS = (
    "@env.initialize\nasync def start():\n"
    '    Path(f"events/{os.getpid()}").write_text("started")\n'
    "@env.shutdown\nasync def stop():\n"
    '    Path(f"events/{os.getpid()}").write_text("stopped")\n'
)
ONE_FILE = IMPORTS + ENV + SOLVE + HOOKS
TASKS = "tasks = [solve()]\n"
FROM_ENV = "from env import solve\n" + TASKS


LAYOUTS: dict[str, tuple[dict[str, str], str]] = {
    "one-file": ({"env.py": ONE_FILE + TASKS}, "env.py"),
    "tasks-beside-the-env": ({"env.py": ONE_FILE, "tasks.py": FROM_ENV}, "tasks.py"),
    "json-rows": (
        {"env.py": ONE_FILE, "tasks.json": json.dumps([{"env": "local-test", "id": "solve"}])},
        "tasks.json",
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_project_layout_runs_in_fresh_env_processes(
    hud: Hud, gateway: Models, tmp_path: Path, layout: str
) -> None:
    files, source = LAYOUTS[layout]
    project = tmp_path / "project"
    write_files(project, {**files, "asset.txt": "ok", "events/": ""})

    document = run(
        hud, str(project / source), *SCRIPTED, "--all", "--group", "2", "--max-concurrent", "2"
    )

    assert (document["run_count"], document["mean_reward"], document["error_count"]) == (2, 1.0, 0)
    events = list((project / "events").iterdir())
    assert [event.read_text() for event in events] == ["stopped", "stopped"]


@pytest.mark.parametrize(
    ("source", "same_name", "errors"),
    [
        ("rows.json", False, 0),
        (".", True, 1),
    ],
)
def test_a_row_runs_in_the_environment_it_names(
    hud: Hud, gateway: Models, tmp_path: Path, source: str, same_name: bool, errors: int
) -> None:
    """Rows resolve their env by name; two envs sharing that name make the row an error."""
    project = tmp_path / "project"
    template = '@env.template(id="solve")\nasync def solve():\n    answer = yield "answer ok"\n'
    write_files(
        project,
        {
            "bound_env.py": 'from hud import Environment\nenv = Environment("bound")\n'
            + template
            + '    yield float(answer == "ok")\n',
            "env.py": "from hud import Environment\n"
            f"env = Environment({'bound' if same_name else 'unrelated'!r})\n"
            + template
            + "    yield 0.0\n",
            "tasks.py": "from bound_env import solve\ntasks = [solve()]\n",
            "rows.json": '{"env": "bound", "id": "solve"}\n',
        },
    )

    result = hud("eval", str(project / source), *SCRIPTED, "--all", "--yes", "--json")

    assert result.exit_code == errors, result
    assert (result.json["run_count"], result.json["error_count"]) == (1, errors)


def test_a_nested_verifier_runs_in_its_own_environment(
    hud: Hud, gateway: Models, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    write_files(
        project,
        {
            "core.py": 'from hud import Environment\nactor = Environment("actor")\n'
            + 'judge = Environment("judge")\n',
            "actor.py": "from core import actor\n"
            '@actor.template(id="solve")\nasync def solve():\n'
            '    answer = yield "answer ok"\n    yield {"score": 0.0, "answer": answer}\n',
            "judge.py": "from core import judge\n"
            '@judge.template(id="verify")\nasync def verify():\n'
            '    result = yield ""\n    yield 1.0 if result["answer"] == "ok" else 0.0\n',
            "env.py": "from core import actor, judge\n"
            "from actor import solve\nfrom judge import verify\n",
            "tasks.py": "from env import solve, verify\n"
            "tasks = [solve()]\ntasks[0].verifier = verify()\n",
        },
    )

    document = run(hud, str(project / "tasks.py"), *SCRIPTED)

    assert (document["run_count"], document["mean_reward"]) == (1, 1.0)


async def test_eval_attaches_to_an_already_served_environment(hud: Hud, gateway: Models) -> None:
    env = Environment("served")

    @env.template(id="solve")
    async def solve():
        answer = yield "answer ok"
        yield 1.0 if answer == "ok" else 0.0

    (hud.cwd / "rows.json").write_text('[{"env": "served", "id": "solve"}]')
    async with LocalRuntime(env)(Task(env="served", id="solve")) as runtime:
        document = await asyncio.to_thread(
            run, hud, "rows.json", *SCRIPTED, "--runtime", runtime.url
        )

    assert (document["run_count"], document["mean_reward"]) == (1, 1.0)


def test_one_tasks_file_is_picked_when_no_source_is_given(hud: Hud, gateway: Models) -> None:
    write_files(
        hud.cwd,
        {
            "env.py": TASKS_PY,
            "rows.json": '[{"env": "demo", "id": "solve", "args": {"n": 4}}]',
            ".hidden.json": "[]",
            ".hud_eval.toml": '[eval]\nagent = "openai_compatible"\nmodel = "m"\n',
        },
    )

    document = run(hud)

    assert (document["source"], document["mean_reward"]) == ("rows.json", 1.0)
    assert [request.prompt for request in gateway.requests()] == ["solve 4"]


def test_a_python_source_is_imported_on_the_main_thread(hud: Hud, gateway: Models) -> None:
    (hud.cwd / "probe.py").write_text(
        "import threading\nfrom pathlib import Path\n"
        'Path("main-thread.txt").write_text('
        "str(threading.current_thread() is threading.main_thread()))\n"
        "tasks = []\n"
    )

    result = hud("eval", "probe.py", "openai", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert (hud.cwd / "main-thread.txt").read_text() == "True"


# ─── planning and refusals ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "argv", "expected"),
    [
        pytest.param(
            "", ["tasks.py", "openai"], {"runtime": "local"}, id="a-local-source-runs-here"
        ),
        pytest.param(
            "", ["My Tasks", "openai"], {"runtime": "hosted"}, id="a-platform-taskset-runs-hosted"
        ),
        pytest.param(
            "[eval]\nremote = true\n",
            ["tasks.py", "openai"],
            {"runtime": "hosted"},
            id="the-toml-asks-for-hosted",
        ),
        pytest.param(
            "[eval]\nremote = false\n",
            ["tasks.py", "openai", "--remote"],
            {"runtime": "hosted"},
            id="a-flag-beats-the-toml",
        ),
        pytest.param(
            '[openai]\nmodel = "${MY_EVAL_MODEL}"\n',
            ["tasks.py", "openai"],
            {"model": "gpt-4o"},
            id="the-toml-model-from-a-variable",
        ),
    ],
)
def test_a_dry_run_reports_the_plan_and_runs_nothing(
    hud: Hud,
    platform: Platform,
    hud_env: HudEnv,
    tasks: Path,
    config: str,
    argv: list[str],
    expected: dict[str, Any],
) -> None:
    hud_env.set(MY_EVAL_MODEL="gpt-4o")
    (hud.cwd / ".hud_eval.toml").write_text(config)

    result = hud("eval", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert {name: result.json[name] for name in expected} == expected
    assert platform.services.requests() == []
