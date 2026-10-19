"""What a platform command asks the platform for, on the object its reference names.

A reference is a name, a display name, an id, or an id in another spelling (upper
case, no dashes); each row runs one command and reads back what it acted on: a
platform record, the request it sent, the directory link, or the answer it printed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply
from tests.support.platform import BROWSER_PROJECT_ID

from .commands import (
    BASE_MODEL_ID,
    ENV_ID,
    IMPORT_ID,
    JOB_ID,
    MODEL_ID,
    TB2_ID,
    TRACE_ID,
    Setup,
    a_base_model,
    a_job,
    a_registered_environment,
    a_running_import,
    a_trainable_model,
    nothing,
    tasks,
    together,
)

if TYPE_CHECKING:
    from tests.support import Hud, Platform
    from tests.support.machine import Result

TASK_ID = "00000000-0000-4000-a000-000000000005"

Observe = Callable[["Hud", "Platform", "Result"], Any]


def printed(*path: str) -> Observe:
    def observe(hud: Hud, platform: Platform, result: Result) -> Any:
        found = result.json
        for key in path:
            found = (found or {}).get(key)
        return found

    return observe


def linked(key: str) -> Observe:
    def observe(hud: Hud, platform: Platform, result: Result) -> Any:
        return json.loads((hud.cwd / ".hud" / "config.json").read_text()).get(key)

    return observe


def asked(path: str, *keys: str) -> Observe:
    """The ``keys`` of the query the last GET of ``path`` carried."""

    def observe(hud: Hud, platform: Platform, result: Result) -> Any:
        query = platform.services.requests("api", "GET", path)[-1].query
        return {key: query.get(key) for key in keys}

    return observe


def asked_for(path: str) -> Observe:
    """The path parameters of the last GET of ``path``."""

    def observe(hud: Hud, platform: Platform, result: Result) -> Any:
        return platform.services.requests("api", "GET", path)[-1].params

    return observe


def cancellations(hud: Hud, platform: Platform, result: Result) -> Any:
    return platform.cancellations


def heads(hud: Hud, platform: Platform, result: Result) -> Any:
    return platform.heads


def forked_from(hud: Hud, platform: Platform, result: Result) -> Any:
    return [model["forked_from"] for model in platform.models if "forked_from" in model]


def import_cancelled(hud: Hud, platform: Platform, result: Result) -> Any:
    return platform.imports[IMPORT_ID].cancelled


def created_project(hud: Hud, platform: Platform, result: Result) -> Any:
    (created,) = [project for project in platform.projects if project["name"] == "UI Evals"]
    link = hud.cwd / ".hud" / "config.json"
    linked_to = json.loads(link.read_text())["project_id"] if link.exists() else None
    return (result.json["id"] == created["id"], created.get("description"), linked_to is not None)


def a_finished_import(hud: Hud, platform: Platform) -> None:
    a_running_import(hud, platform)
    platform.imports[IMPORT_ID].statuses = ["completed"]


def environments_sharing_a_prefix(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_environment("browser")
    platform.add_environment("browser-anchor", id=ENV_ID)


def a_demo_taskset(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_taskset("demo")


def the_fork_name_is_taken(hud: Hud, platform: Platform) -> None:
    a_base_model(hud, platform)
    a_trainable_model(hud, platform)
    platform.services.route(
        "api", "POST", "/v2/models/fork", Reply(status=409, json={"detail": "Model exists"})
    )


@dataclass(frozen=True)
class Effect:
    argv: list[str]
    observe: Observe
    expected: Any
    setup: Setup = nothing


EFFECTS: dict[str, Effect] = {
    "jobs get: a job id without dashes": Effect(
        ["jobs", JOB_ID.replace("-", "")],
        asked_for("/v2/jobs/{id}/traces"),
        {"id": JOB_ID},
        a_job,
    ),
    "jobs cancel: a job": Effect(
        ["jobs", "cancel", JOB_ID, "--yes"], cancellations, [("job", {"job_id": JOB_ID})]
    ),
    "jobs cancel: one trace": Effect(
        ["jobs", "cancel", JOB_ID, "--trace-id", TRACE_ID, "--yes"],
        cancellations,
        [("trace", {"job_id": JOB_ID, "trace_id": TRACE_ID})],
    ),
    "jobs cancel: every job": Effect(
        ["jobs", "cancel", "--all", "--yes"], cancellations, [("all", {})]
    ),
    "models head: by display name": Effect(
        ["models", "head", "owner/model + version"],
        lambda hud, platform, result: (result.json["model_id"], result.json["head"]["id"]),
        (MODEL_ID, "ckpt-2"),
        a_trainable_model,
    ),
    "models head: by id": Effect(
        ["models", "head", MODEL_ID],
        lambda hud, platform, result: (result.json["model_id"], result.json["head"]["id"]),
        (MODEL_ID, "ckpt-2"),
        a_trainable_model,
    ),
    "models head: a model without checkpoints": Effect(
        ["models", "head", "claude-sonnet-4-6"], printed("head"), None, a_base_model
    ),
    "models head --set: by short name": Effect(
        ["models", "head", "my-sonnet", "--set", "ckpt-1", "--yes"],
        heads,
        {MODEL_ID: "ckpt-1"},
        a_trainable_model,
    ),
    "models fork: by display name": Effect(
        ["models", "fork", "Claude Sonnet", "--name", "fresh", "--yes"],
        forked_from,
        [BASE_MODEL_ID],
        a_base_model,
    ),
    "models fork --if-not-exists: onto a taken name": Effect(
        ["models", "fork", "Claude Sonnet", "--name", "my-sonnet", "--if-not-exists", "--yes"],
        lambda hud, platform, result: (result.json["id"], result.json["existed"]),
        (MODEL_ID, True),
        the_fork_name_is_taken,
    ),
    "project create: links the directory": Effect(
        ["project", "create", "UI Evals", "--description", "UI tasks", "--yes"],
        created_project,
        (True, "UI tasks", True),
    ),
    "project create --no-use": Effect(
        ["project", "create", "UI Evals", "--no-use", "--yes"], created_project, (True, None, False)
    ),
    "project use: by display name": Effect(
        ["project", "use", "Browser Evals"], linked("project_id"), BROWSER_PROJECT_ID
    ),
    "project use: by an upper-case id": Effect(
        ["project", "use", BROWSER_PROJECT_ID.upper()], linked("project_id"), BROWSER_PROJECT_ID
    ),
    "project list: marks the default and read-only projects": Effect(
        ["project", "list"],
        lambda hud, platform, result: [
            (p["name"], p["is_default"], p["can_create"]) for p in result.json
        ],
        [("default", True, True), ("browser-evals", False, True), ("locked-down", False, False)],
    ),
    "sync env: by name across separators": Effect(
        ["sync", "env", "browser_anchor"],
        linked("registry_id"),
        ENV_ID,
        environments_sharing_a_prefix,
    ),
    "sync env: by id": Effect(
        ["sync", "env", ENV_ID], linked("registry_id"), ENV_ID, a_registered_environment
    ),
    "harbor imports: a taskset by name": Effect(
        ["harbor", "imports", "--taskset", "tb2"],
        asked("/v2/imports", "taskset_id", "format"),
        {"taskset_id": [TB2_ID], "format": ["harbor"]},
        a_running_import,
    ),
    "harbor cancel: a live import": Effect(
        ["harbor", "cancel", IMPORT_ID, "--yes"], import_cancelled, True, a_running_import
    ),
    "harbor cancel: an ended import": Effect(
        ["harbor", "cancel", IMPORT_ID, "--yes"], import_cancelled, False, a_finished_import
    ),
    "sync tasks --project: places the upload without linking it": Effect(
        ["sync", "tasks", "demo", "tasks.py", "--project", BROWSER_PROJECT_ID, "--yes"],
        lambda hud, platform, result: (
            [upload["project_id"] for upload in platform.uploads],
            (hud.cwd / ".hud" / "config.json").exists(),
        ),
        ([BROWSER_PROJECT_ID], False),
        together(tasks, a_demo_taskset),
    ),
    "qa results: by task": Effect(
        ["qa", "results", "--level", "task", TASK_ID],
        asked("/v2/qa/results", "task_ids"),
        {"task_ids": [TASK_ID]},
    ),
}


@pytest.mark.parametrize("case", EFFECTS)
def test_a_command_acts_on_the_object_its_reference_names(
    hud: Hud, platform: Platform, case: str
) -> None:
    effect = EFFECTS[case]
    effect.setup(hud, platform)

    result = hud(*effect.argv, "--json")

    assert result.exit_code == 0, result
    assert effect.observe(hud, platform, result) == effect.expected
