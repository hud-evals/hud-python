"""``hud eval``: run real rollouts of a taskset with a scripted model, where each row says."""

from __future__ import annotations

import json
import os
import signal
import time
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import write_files
from tests.fixtures.models import Turn
from tests.fixtures.services import Reply

from .conftest import API_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Hud, HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

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
def gateway(models: Models, signed_in: FakeServices) -> Models:
    """A signed-in user whose agents reach a scripted model that always answers "ok"."""
    models.script([Turn(text="ok")])
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
    ("grade", "selection", "exit_code", "rewards", "errors"),
    [
        pytest.param(
            "0.0", ["--all"], 0, [1.0, 0.0], [False, False], id="a-low-reward-is-not-an-error"
        ),
        pytest.param(
            '{"score": 0.0, "isError": True, "content": "grader unavailable"}',
            ["--all"],
            0,
            [1.0, 0.0],
            [False, True],
            id="a-grade-error",
        ),
        pytest.param("1 / 0", ["--all"], 0, [1.0, 0.0], [False, True], id="a-grader-crash"),
        pytest.param("1 / 0", ["--task-ids", "1"], 1, [0.0], [True], id="no-run-graded"),
    ],
)
def test_eval_reports_every_run_and_fails_when_none_was_graded(
    hud: Hud,
    gateway: Models,
    grade: str,
    selection: list[str],
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

    result = hud("eval", "tasks.py", *SCRIPTED, *selection, "--yes", "--json")

    assert result.exit_code == exit_code, result
    runs = sorted(result.json["runs"], key=lambda run: run["is_error"])
    assert [run["reward"] for run in runs] == rewards
    assert [run["is_error"] for run in runs] == errors
    assert result.json["error_count"] == sum(errors)


def test_eval_grades_a_run_whose_model_failed_and_reports_the_failure(
    hud: Hud, models: Models, signed_in: FakeServices, tasks: Path
) -> None:
    models.script([Turn(fail=Reply(status=400, json={"error": {"message": "model unavailable"}}))])

    document = run(hud, "tasks.py", *SCRIPTED)

    (failed,) = document["runs"]
    assert (failed["reward"], failed["is_error"], document["error_count"]) == (0.0, False, 0)
    assert "model unavailable" in failed["trace_error"]


@pytest.mark.parametrize(
    ("argv", "prompts"),
    [
        pytest.param([], ["solve 0"], id="the-first-by-default"),
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
            ["openai_compatible", "-m", "custom", "-c", "api_key=custom-key", "-c", "base_url=URL"],
            "",
            {"key": "custom-key"},
            id="custom-endpoint",
        ),
    ],
)
def test_the_agent_reaches_its_model_with_the_right_key_and_settings(
    hud: Hud,
    signed_in: FakeServices,
    gateway: Models,
    hud_env: HudEnv,
    tasks: Path,
    argv: list[str],
    config: str,
    expected: dict[str, Any],
) -> None:
    """Provider SDKs read their own ``*_BASE_URL``, which here points at the scripted model."""
    signed_in.route("api", "GET", "/v2/models", json={"items": [GLM], "total": 1})
    (hud.cwd / ".hud_eval.toml").write_text(config)
    hud_env.set(
        OPENAI_BASE_URL=gateway.url,
        ANTHROPIC_BASE_URL=gateway.url,
        MY_EVAL_MODEL="z-ai/glm-5.2-from-env",
    )

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
    }
    assert {name: observed[name] for name in expected} == expected


# ─── project layouts ────────────────────────────────────────────────────

ENV = 'from hud import Environment\nenv = Environment("local-test")\n'
SOLVE = (
    '@env.template(id="solve")\nasync def solve():\n'
    '    answer = yield "answer ok"\n'
    '    yield 1.0 if answer == "ok" else 0.0\n'
)


def named(name: str, reward: str) -> str:
    """An env ``name`` whose ``solve`` pays ``reward``."""
    return (
        f"from hud import Environment\nenv = Environment({name!r})\n"
        '@env.template(id="solve")\nasync def solve():\n'
        f'    answer = yield "answer ok"\n    yield {reward}\n'
    )


LAYOUTS: dict[str, tuple[dict[str, str], str, tuple[int, float, int]]] = {
    "tasks beside the env": (
        {"env.py": ENV + SOLVE, "tasks.py": "from env import solve\ntasks = [solve()]\n"},
        "tasks.py",
        (1, 1.0, 0),
    ),
    "json rows beside the env": (
        {"env.py": ENV + SOLVE, "tasks.json": json.dumps([{"env": "local-test", "id": "solve"}])},
        "tasks.json",
        (1, 1.0, 0),
    ),
    "a row picks its env by name among several": (
        {
            "bound_env.py": named("bound", 'float(answer == "ok")'),
            "env.py": named("unrelated", "0.0"),
            "rows.json": '{"env": "bound", "id": "solve"}\n',
        },
        "rows.json",
        (1, 1.0, 0),
    ),
    "two envs sharing the row's name error the row": (
        {
            "bound_env.py": named("bound", 'float(answer == "ok")'),
            "env.py": named("bound", "0.0"),
            "tasks.py": "from bound_env import solve\ntasks = [solve()]\n",
        },
        ".",
        (1, 0.0, 1),
    ),
    "a verifier in another env of the project": (
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
        "tasks.py",
        (1, 1.0, 0),
    ),
}


@pytest.mark.parametrize(("files", "source", "outcome"), LAYOUTS.values(), ids=LAYOUTS.keys())
def test_eval_runs_a_projects_rows_in_the_environments_they_name(
    hud: Hud,
    gateway: Models,
    tmp_path: Path,
    files: dict[str, str],
    source: str,
    outcome: tuple[int, float, int],
) -> None:
    project = tmp_path / "project"
    write_files(project, files)

    result = hud("eval", str(project / source), *SCRIPTED, "--all", "--yes", "--json")

    assert result.exit_code == (outcome[2] > 0), result
    document = result.json
    assert (document["run_count"], document["mean_reward"], document["error_count"]) == outcome


# ─── planning and refusals ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "argv", "hosted"),
    [
        pytest.param("", ["My Tasks"], True, id="a-platform-taskset-runs-hosted"),
        pytest.param("[eval]\nremote = true\n", ["tasks.py"], True, id="the-toml-asks-for-hosted"),
        pytest.param(
            "[eval]\nremote = false\n", ["tasks.py", "--remote"], True, id="a-flag-beats-the-toml"
        ),
    ],
)
def test_eval_runs_each_row_where_its_placement_says(
    hud: Hud,
    gateway: Models,
    signed_in: FakeServices,
    tasks: Path,
    config: str,
    argv: list[str],
    hosted: bool,
) -> None:
    signed_in.route(
        "api", "GET", "/v2/tasksets/by-name/My Tasks", json={"taskset_id": "ts", "name": "My Tasks"}
    )
    signed_in.route(
        "api",
        "GET",
        "/v2/tasksets/ts/export",
        json={"name": "My Tasks", "tasks": [{"name": "one", "env": "demo", "scenario": "solve"}]},
    )
    signed_in.route("api", "POST", "/v2/rollouts/submit", json={"status": "queued"})
    signed_in.route("api", "GET", "/v2/trace/{id}", json={"status": "completed", "reward": 0.5})
    (hud.cwd / ".hud_eval.toml").write_text(config)

    document = run(hud, argv[0], *SCRIPTED, *argv[1:])

    assert document["mean_reward"] == (0.5 if hosted else 1.0)
    assert len(signed_in.requests("api", "POST", "/v2/rollouts/submit")) == hosted
    assert [request.prompt for request in gateway.requests()] == ([] if hosted else ["solve 0"])


# ─── interruption ───────────────────────────────────────────────────────


def test_interrupting_a_hosted_eval_cancels_every_rollout_on_the_platform(
    hud: Hud, signed_in: FakeServices, tasks: Path
) -> None:
    signed_in.route("api", "POST", "/v2/rollouts/submit", json={"status": "queued"})
    signed_in.route("api", "GET", "/v2/trace/{id}", json={"status": "running"})
    signed_in.route("api", "POST", "/v2/rollouts/cancel", json={"status": "accepted"})
    process = hud.start("eval", "tasks.py", *SCRIPTED, "--remote", "--group-size", "3", "--yes")

    deadline = time.monotonic() + 60
    while len(signed_in.requests("api", "GET", "/v2/trace/{id}")) < 3:
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "hud eval never polled its rollouts"
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)

    submitted = {
        body["trace_id"] for body in signed_in.bodies("api", "POST", "/v2/rollouts/submit")
    }
    cancelled = [
        body["trace_id"] for body in signed_in.bodies("api", "POST", "/v2/rollouts/cancel")
    ]
    assert process.returncode == 130
    assert len(submitted) == 3
    assert sorted(cancelled) == sorted(submitted)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_terminating_eval_tears_down_the_environments_it_spawned(
    hud: Hud, models: Models, signed_in: FakeServices
) -> None:
    pids = hud.cwd / "env.pids"
    (hud.cwd / "tasks.py").write_text(
        f"import os\n{TASKS_PY}\n\n@env.initialize\nasync def record_pid() -> None:\n"
        f"    with open({str(pids)!r}, 'a') as out:\n"
        "        out.write(f'{os.getpid()}\\n')\n"
    )
    models.script([Turn(fail=Reply(json={}, delay=600))])
    process = hud.start("eval", "tasks.py", *SCRIPTED, "--group-size", "2", "--yes")

    try:
        deadline = time.monotonic() + 60
        while len(models.requests()) < 2:
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "hud eval never reached the model"
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=30)
    finally:
        process.kill()
        spawned = [int(pid) for pid in pids.read_text().split()] if pids.exists() else []
        running = [pid for pid in spawned if alive(pid)]
        for pid in running:
            os.kill(pid, signal.SIGKILL)

    assert len(spawned) == 2
    assert running == []
    assert process.returncode == 128 + signal.SIGTERM
