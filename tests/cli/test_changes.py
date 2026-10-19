"""What a ``hud`` command may change: nothing until confirmed, nothing on a dry run, and
nothing when it fails.

State is every file under the working directory and the user's ``~/.hud``, and every
platform request that could write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pytest

from tests.support.platform import (
    BROWSER_PROJECT_ID,
    LOCKED_PROJECT_ID,
    TEAM_ID,
    USER_ID,
    Import,
)

from .commands import (
    MUTATING,
    PLATFORM,
    TRACE_ID,
    Setup,
    a_base_model,
    a_platform_taskset,
    a_registered_environment,
    environment,
    files,
    harbor_folders,
    link,
    linked_taskset_with_a_change,
    nothing,
    refused,
    state,
    tasks,
    together,
    writes,
)

if TYPE_CHECKING:
    from tests.support import Hud, HudEnv, Platform


@pytest.mark.parametrize("name", sorted(MUTATING))
@pytest.mark.parametrize("flag", ["", "--dry-run"], ids=["unconfirmed", "dry-run"])
def test_a_mutating_command_changes_nothing_unless_confirmed(
    hud: Hud, platform: Platform, hud_env: HudEnv, name: str, flag: str
) -> None:
    command = PLATFORM[name]
    command.setup(hud, platform)
    before, sent = state(hud, hud_env), len(writes(platform))

    result = hud(*command.argv, *filter(None, [flag]), "--json")

    if flag:
        assert (result.exit_code, result.json["dry_run"]) == (0, True), result
    else:
        assert (result.exit_code, result.json["error"]) == (2, "usage"), result
        assert "--yes" in result.json["suggestion"]
    assert writes(platform)[sent:] == []
    assert state(hud, hud_env) == before


# ─── failures ───────────────────────────────────────────────────────────

OTHER_ID = "cccccccc-0000-4000-8000-000000000001"
EXAMPLE_ENV = 'from hud import Environment\nenv = Environment("example")\n'
HARBOR_TASK = {"task.toml": "version = '1.0'\n", "environment/Dockerfile": "FROM ubuntu\n"}


def scoped(user: str, team: str, *, elsewhere: bool = False) -> Setup:
    """A link made by ``user`` in ``team``, on this platform or another one."""

    def setup(hud: Hud, platform: Platform) -> None:
        origin = "https://api.example" if elsewhere else platform.services.url("api")
        scope = {"origin": origin.removesuffix("/api"), "user_id": user, "team_id": team}
        link({"version": 1, "scope": scope, "project_id": BROWSER_PROJECT_ID})(hud, platform)

    return setup


def uneditable_taskset(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_taskset("theirs", can_edit=False)


def small_import_limit(hud: Hud, platform: Platform) -> None:
    del hud
    platform.import_formats[0]["max_input_bytes"] = 10


def an_import_of_another_format(hud: Hud, platform: Platform) -> None:
    del hud
    platform.imports["imp"] = Import("imp", "tb2", format="parquet")


def trace_and_task_checks(hud: Hud, platform: Platform) -> None:
    del hud
    platform.qa_checks = [
        {"key": "failure_analysis", "subject_type": "trace", "title": "FA"},
        {"key": "task_reward_exploit", "subject_type": "task", "title": "TRE"},
    ]


def a_deleted_taskset(hud: Hud, platform: Platform) -> None:
    tasks(hud, platform)
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    platform.tasksets.clear()


def demo_taskset(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_taskset("demo")


@dataclass(frozen=True)
class Failure:
    """An invocation that fails, and what it may have asked the platform before failing.

    ``reaches`` is ``nothing`` (no request at all), ``reads`` (no request that could
    write) or ``the refusal`` (the platform refused the write the command sent). Unless
    ``linked`` is false the directory starts linked to a project, so a failure that
    rewrote the link would show.
    """

    argv: list[str]
    setup: Setup = nothing
    exit_code: int = 1
    error: str = "failure"
    reaches: Literal["nothing", "reads", "the refusal"] = "reads"
    env: dict[str, str | None] = field(default_factory=dict)
    mentions: str = ""
    linked: bool = True


FAILURES: dict[str, Failure] = {
    "deploy: no environment declared": Failure(["deploy", "--yes"], exit_code=2, error="usage"),
    "deploy: several environments and no name": Failure(
        ["deploy", "--yes"],
        files({"a.py": 'a = Environment("a")\n', "b.py": 'b = Environment("b")\n'}),
        exit_code=2,
        error="usage",
    ),
    "deploy: a read-only project": Failure(
        ["deploy", "--project", LOCKED_PROJECT_ID, "--yes"],
        environment,
        error="permission_denied",
    ),
    "deploy: the build is refused": Failure(
        ["deploy", "--no-env", "--yes"],
        together(environment, refused("POST", "/v2/builds/trigger", 409)),
        error="conflict",
        reaches="the refusal",
    ),
    "eval: an unknown setting": Failure(
        ["eval", "tasks.py", "openai", "--yes"],
        together(tasks, files({".hud_eval.toml": '[eval]\nmodle = "x"\n'})),
        exit_code=2,
        error="usage",
        reaches="nothing",
    ),
    "eval: an unset variable": Failure(
        ["eval", "tasks.py", "openai", "--yes"],
        together(tasks, files({".hud_eval.toml": '[openai]\nmodel = "${HUD_EVAL_UNSET}"\n'})),
        exit_code=2,
        error="usage",
        reaches="nothing",
    ),
    "eval: no source found": Failure(["eval", "--yes"], error="not_found", reaches="nothing"),
    "eval: the hud runtime signed out": Failure(
        ["eval", "tasks.py", "openai", "--runtime", "hud", "--yes"],
        tasks,
        error="permission_denied",
        reaches="nothing",
        env={"HUD_API_KEY": None},
    ),
    "eval: a platform taskset without container rows run locally": Failure(
        ["eval", "Demo", "openai", "--runtime", "local", "--yes"],
        a_platform_taskset,
        exit_code=2,
        error="usage",
    ),
    "init: a non-empty directory": Failure(
        ["init", "taken", "--preset", "blank"],
        files({"taken/precious.txt": "data"}),
        error="conflict",
        reaches="nothing",
    ),
    "init: a symlink inside the target": Failure(
        ["init", "project", "--preset", "blank", "--force"],
        files({"outside.py": "original", "project/env.py": "->outside.py"}),
        reaches="nothing",
    ),
    "init: a symlinked target": Failure(
        ["init", "project", "--preset", "blank"],
        files({"outside/": "", "project": "->outside"}),
        reaches="nothing",
    ),
    "harbor import: a taskset the caller cannot edit": Failure(
        ["harbor", "import", "tasks", "run-1", "--taskset", "theirs", "--yes"],
        together(harbor_folders, uneditable_taskset),
        error="permission_denied",
    ),
    "harbor import: a bundle over the platform limit": Failure(
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--yes"],
        together(harbor_folders, small_import_limit),
        exit_code=2,
        error="usage",
    ),
    "harbor import: two folders with one name": Failure(
        ["harbor", "import", "a/hello", "b/hello", "--yes"],
        files(
            {f"{root}/hello/{name}": text for root in "ab" for name, text in HARBOR_TASK.items()}
        ),
        exit_code=2,
        error="usage",
    ),
    "harbor import: the taskset is refused": Failure(
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--link", "--yes"],
        together(harbor_folders, refused("POST", "/v2/tasksets", 403)),
        error="permission_denied",
        reaches="the refusal",
    ),
    "harbor report: an import of another format": Failure(
        ["harbor", "report", "imp"], an_import_of_another_format, exit_code=2, error="usage"
    ),
    "sync: nothing linked": Failure(["sync"], tasks, exit_code=2, error="usage"),
    "sync: a missing source": Failure(
        ["sync", "tasks", "demo", "missing.json", "--yes"], error="not_found"
    ),
    "sync: a read-only project": Failure(
        ["sync", "tasks", "demo", "tasks.py", "--project", LOCKED_PROJECT_ID, "--yes"],
        together(tasks, demo_taskset),
        error="permission_denied",
    ),
    "sync: a stored taskset that was deleted": Failure(
        ["sync", "tasks", "--force", "--yes"], a_deleted_taskset, error="not_found"
    ),
    "sync: the upload is refused": Failure(
        ["sync", "--yes"],
        together(linked_taskset_with_a_change, refused("POST", "/v2/tasks/upload", 400)),
        reaches="the refusal",
    ),
    "sync tasks: the upload is refused": Failure(
        ["sync", "tasks", "demo", "tasks.py", "--yes"],
        together(tasks, refused("POST", "/v2/tasks/upload", 400)),
        reaches="the refusal",
    ),
    "sync env: an unknown environment": Failure(
        ["sync", "env", "bbbbbbbb-0000-4000-8000-999999999999"],
        a_registered_environment,
        error="not_found",
    ),
    "jobs cancel: no target": Failure(
        ["jobs", "cancel", "--yes"], exit_code=2, error="usage", reaches="nothing"
    ),
    "qa run: trace and task checks together": Failure(
        ["qa", "run", "failure_analysis,task_reward_exploit", TRACE_ID, "--yes"],
        trace_and_task_checks,
        exit_code=2,
        error="usage",
    ),
    "task start: an id several rows share": Failure(
        ["task", "start", "solve", "--source", "tasks.json"],
        files(
            {
                "env.py": EXAMPLE_ENV + '@env.template(id="solve")\nasync def solve():\n'
                '    yield "go"\n    yield 1.0\n',
                "tasks.json": json.dumps(
                    [{"env": "example", "id": "solve", "slug": s} for s in ("one", "two")]
                ),
            }
        ),
        exit_code=2,
        error="usage",
        reaches="nothing",
    ),
    "models fork: a misspelt model": Failure(
        ["models", "fork", "claude-sonet-4-6", "--name", "mine", "--yes"],
        a_base_model,
        exit_code=2,
        error="usage",
        mentions="claude-sonnet-4-6",
    ),
    "project: a default project that does not exist": Failure(
        ["project"], error="not_found", env={"HUD_DEFAULT_PROJECT": "nope"}, linked=False
    ),
    "project create: the platform refuses": Failure(
        ["project", "create", "UI Evals", "--yes"],
        refused("POST", "/v2/projects", 403),
        error="permission_denied",
        reaches="the refusal",
    ),
    "project use: a read-only project": Failure(
        ["project", "use", "locked-down"], error="permission_denied"
    ),
    "project use: a link made by another user": Failure(
        ["project", "use", "browser-evals"], scoped(OTHER_ID, TEAM_ID), reaches="reads"
    ),
    "project use: a link made in another team": Failure(
        ["project", "use", "browser-evals"], scoped(USER_ID, OTHER_ID), reaches="reads"
    ),
    "project use: a link made on another platform": Failure(
        ["project", "use", "browser-evals"],
        scoped(USER_ID, TEAM_ID, elsewhere=True),
        reaches="reads",
    ),
    "project use: a corrupt link": Failure(
        ["project", "use", "browser-evals"], link('{"broken":'), reaches="reads"
    ),
}


@pytest.mark.parametrize("case", FAILURES)
def test_a_command_that_fails_changes_nothing(
    hud: Hud, platform: Platform, hud_env: HudEnv, case: str
) -> None:
    failure = FAILURES[case]
    if failure.linked:
        link({"project_id": BROWSER_PROJECT_ID})(hud, platform)
    failure.setup(hud, platform)
    hud_env.set(**failure.env)
    before, asked = state(hud, hud_env), len(platform.services.requests())

    result = hud(*failure.argv, "--json")

    assert (result.exit_code, result.json["error"]) == (failure.exit_code, failure.error), result
    assert failure.mentions in result.json["message"]
    assert state(hud, hud_env) == before
    sent = [r for r in platform.services.requests()[asked:] if r.service != "telemetry"]
    if failure.reaches == "nothing":
        assert sent == []
    elif failure.reaches == "reads":
        assert [f"{r.method} {r.path}" for r in sent if r.method != "GET"] == []
