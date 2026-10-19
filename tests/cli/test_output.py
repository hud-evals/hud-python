"""What a ``hud`` command prints: one error document when it fails, the same error kind for
each platform refusal, one id per line when quiet, and never a secret it was given."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import Reply, write_files
from tests.support.platform import Import

from .commands import (
    EVERY_COMMAND,
    JOB_ID,
    PLATFORM,
    TREE,
    Setup,
    a_trainable_model,
    answer_everything,
    environment,
    failing,
    harbor_folders,
    invocation,
    refused,
    tasks,
    together,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.support import Hud, HudEnv, Platform


@pytest.mark.parametrize("name", EVERY_COMMAND)
def test_a_failure_is_one_error_document_under_json_and_stays_off_stdout_in_text(
    hud: Hud, platform: Platform, hud_env: HudEnv, name: str
) -> None:
    argv = failing(name, hud, platform, hud_env)

    as_json = hud(*argv, "--json")
    as_text = hud(*argv)

    error = as_json.json
    assert as_json.exit_code == (2 if error["error"] == "usage" else 1), as_json
    assert error["message"] not in as_json.stderr, as_json
    assert (as_text.exit_code, as_text.stdout) == (as_json.exit_code, ""), as_text
    assert error["message"] in as_text.stderr, as_text


ANSWERS = {
    "401": (Reply(status=401, json={"detail": "Refused"}), "permission_denied"),
    "403": (Reply(status=403, json={"detail": "Refused"}), "permission_denied"),
    "404": (Reply(status=404, json={"detail": "Refused"}), "not_found"),
    "409": (Reply(status=409, json={"detail": "Refused"}), "conflict"),
    "429": (Reply(status=429, json={"detail": "Refused"}), "rate_limited"),
    "500": (Reply(status=500, json={"detail": "Refused"}), "server_error"),
    "not-json": (Reply(body="<html>", content_type="text/html"), "failure"),
}

# Every command goes through one status mapping: each command is held to it with one
# status, and the mapping itself is read through one command.
REFUSALS = [(name, "403") for name in sorted(PLATFORM)] + [
    ("jobs get", answer) for answer in ANSWERS if answer != "403"
]


@pytest.mark.parametrize(("name", "answer"), REFUSALS)
def test_a_platform_refusal_is_the_same_error_kind_for_every_command(
    hud: Hud, platform: Platform, name: str, answer: str
) -> None:
    reply, error = ANSWERS[answer]
    PLATFORM[name].setup(hud, platform)
    answer_everything(platform, reply)

    result = hud(*invocation(name), "--json")

    assert (result.exit_code, result.json["error"]) == (1, error), result


def test_a_transient_platform_failure_is_retried_and_the_command_succeeds(
    hud: Hud, platform: Platform
) -> None:
    job = platform.add_job(name="job-0", status="completed")
    listed = Reply(json={"items": list(platform.jobs.values())})
    platform.services.route("api", "GET", "/v2/jobs", Reply(status=503), Reply(status=502), listed)

    result = hud("jobs", "list", "--quiet")

    assert (result.exit_code, result.stdout) == (0, f"{job}\n"), result
    assert len(platform.services.requests("api", "GET", "/v2/jobs")) == 3


# ─── lists ──────────────────────────────────────────────────────────────


def python_tasks(hud: Hud, platform: Platform) -> list[str]:
    del platform
    write_files(
        hud.cwd,
        {
            "env.py": 'from hud import Environment\nenv = Environment("example")\n'
            '@env.template(id="solve")\nasync def solve(target: str = "answer"):\n'
            '    answer = yield f"Say {target}."\n    yield 1.0\n',
            "tasks.py": 'from env import solve\ntasks = [solve(), solve(target="hello")]\n',
        },
    )
    return ["solve", "solve-8da35614"]


def projects(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.projects += [
        {"id": f"dddddddd-0000-4000-8000-{n:012d}", "name": f"p{n}"} for n in range(3)
    ]
    return [project["id"] for project in platform.projects]


def catalog(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.add_model(name="owner/model + version", model_name="team/my-sonnet")
    platform.add_model(name="Claude Sonnet", model_name="claude-sonnet-4-6")
    return ["claude-sonnet-4-6", "team/my-sonnet"]


def checkpoints(hud: Hud, platform: Platform) -> list[str]:
    a_trainable_model(hud, platform)
    return ["ckpt-1", "ckpt-2"]


def jobs(hud: Hud, platform: Platform) -> list[str]:
    del hud
    return [platform.add_job(name=f"job-{n}", status="completed") for n in range(3)]


def traces(hud: Hud, platform: Platform) -> list[str]:
    del hud
    ids = [f"00000000-0000-4000-a000-00000000000{n}" for n in range(2)]
    platform.add_job([{"id": trace, "status": "completed"} for trace in ids], id=JOB_ID)
    return ids


CHECKS = [
    {"key": "failure_analysis", "subject_type": "trace", "title": "Failure Analysis"},
    {"key": "reward_hacking", "subject_type": "trace", "title": "Reward Hacking"},
    {"key": "task_reward_exploit", "subject_type": "task", "title": "Task Reward Exploit"},
]


def checks(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.qa_checks = CHECKS
    return [check["key"] for check in CHECKS]


def task_checks(hud: Hud, platform: Platform) -> list[str]:
    checks(hud, platform)
    return ["task_reward_exploit"]


def imports(hud: Hud, platform: Platform) -> list[str]:
    del hud
    taskset = platform.add_taskset("tb2")
    ids = [f"66666666-6666-4666-8666-00000000000{n}" for n in range(2)]
    for found in ids:
        platform.imports[found] = Import(found, taskset)
    return ids


LISTS: dict[str, tuple[list[str], Callable[[Hud, Platform], list[str]]]] = {
    "task list": (["task", "list", "--source", "tasks.py"], python_tasks),
    "project list": (["project", "list"], projects),
    "models list": (["models", "list"], catalog),
    "models checkpoints": (["models", "checkpoints", "team/my-sonnet"], checkpoints),
    "jobs": (["jobs"], jobs),
    "jobs list": (["jobs", "list"], jobs),
    "jobs get": (["jobs", "get", JOB_ID], traces),
    "qa": (["qa"], checks),
    "qa list": (["qa", "list", "--level", "task"], task_checks),
    "harbor imports": (["harbor", "imports", "--taskset", "tb2"], imports),
}


def test_every_command_with_quiet_is_a_list() -> None:
    quiet = {
        name
        for name, command in TREE.items()
        if any("--quiet" in getattr(param, "opts", []) for param in getattr(command, "params", []))
    }

    assert sorted(quiet) == sorted(LISTS)


@pytest.mark.parametrize("name", LISTS)
def test_a_list_prints_one_id_per_line_under_quiet_from_every_page(
    hud: Hud, platform: Platform, name: str
) -> None:
    """The platform pages its project and model catalogs; one item a page makes a list
    of several read every page."""
    argv, seed = LISTS[name]
    expected = seed(hud, platform)
    platform.page_size = 1

    result = hud(*argv, "--quiet")

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (0, expected), result


# ─── secrets ────────────────────────────────────────────────────────────

SECRET = "s3cr3t-c0d3"


def secrets_beside_an_environment(hud: Hud, platform: Platform) -> None:
    environment(hud, platform)
    write_files(hud.cwd, {".env": f"SECRET={SECRET}\n", "token.txt": SECRET})


SECRETS: dict[str, tuple[list[str], Setup | None, dict[str, str], str]] = {
    "set": (["set", f"HUD_API_KEY={SECRET}"], None, {}, SECRET),
    "deploy": (
        [
            *("deploy", "--yes"),
            *("--secret", "id=ENV_TOKEN,env=ENV_TOKEN"),
            *("--secret", "id=FILE_TOKEN,src=token.txt"),
        ],
        secrets_beside_an_environment,
        {"ENV_TOKEN": SECRET},
        SECRET,
    ),
    "eval": (
        [
            *("eval", "tasks.py", "openai_compatible", "-m", "custom"),
            *("-c", f"api_key={SECRET}", "-c", "base_url=http://127.0.0.1:9", "--dry-run"),
        ],
        tasks,
        {},
        SECRET,
    ),
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--yes"],
        harbor_folders,
        {},
        "signature",
    ),
    "harbor import, upload refused": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--yes"],
        together(harbor_folders, refused("PUT", "/uploads/{id}", 403)),
        {},
        "signature",
    ),
}


@pytest.mark.parametrize("name", SECRETS)
@pytest.mark.parametrize("mode", [[], ["--json"]], ids=["text", "json"])
def test_a_secret_given_to_a_command_never_appears_in_its_output(
    hud: Hud, platform: Platform, name: str, mode: list[str]
) -> None:
    argv, setup, env, secret = SECRETS[name]
    if setup is not None:
        setup(hud, platform)

    result = hud(*argv, *mode, env=env)

    assert secret not in result.stdout + result.stderr, result
