"""``hud sync``: push local tasks to a platform taskset, export one, and link a directory to
a deployed environment."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import Task

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
OTHER_ID = "eeeeeeee-0000-4000-8000-000000000002"
ENV_ID = "bbbbbbbb-0000-4000-8000-000000000042"
PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000002"
LOCKED_ID = "aaaaaaaa-0000-4000-8000-000000000003"
TASKS_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=1), solve(n=2)]
"""
ONE, TWO = (Task(env="example", id="solve", args={"n": n}).slug for n in (1, 2))


def exported(*ns: int) -> list[dict[str, Any]]:
    """Tasks as the platform exports them: the local rows for ``solve(n)``."""
    return [
        {
            "name": Task(env="example", id="solve", args={"n": n}).slug,
            "env": "example",
            "scenario": "solve",
            "args": {"n": n},
        }
        for n in ns
    ]


@pytest.fixture
def platform(signed_in: FakeServices, hud: Hud) -> FakeServices:
    """A directory holding ``tasks.py`` and a platform with no ``demo`` taskset yet, which
    creates one on upload."""
    (hud.cwd / "tasks.py").write_text(TASKS_PY)
    signed_in.route("api", "GET", "/v2/tasksets/by-name/{name}", status=404, json={"detail": "No"})
    signed_in.route(
        "api",
        "POST",
        "/v2/tasks/upload",
        json={"taskset_id": TASKSET_ID, "tasks_created": 2, "tasks_updated": 0},
    )
    return signed_in


def holding(
    services: FakeServices, taskset_id: str, name: str, tasks: list[dict[str, Any]]
) -> None:
    """Let the platform hold taskset ``name`` with ``tasks`` under ``taskset_id``."""
    services.route(
        "api", "GET", f"/v2/tasksets/by-name/{name}", json={"taskset_id": taskset_id, "name": name}
    )
    services.route(
        "api", "GET", f"/v2/tasksets/{taskset_id}", json={"id": taskset_id, "name": name}
    )
    services.route(
        "api", "GET", f"/v2/tasksets/{taskset_id}/export", json={"name": name, "tasks": tasks}
    )


def linked(hud: Hud) -> dict[str, Any]:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def link(hud: Hud, **ids: str) -> None:
    (hud.cwd / ".hud").mkdir(exist_ok=True)
    (hud.cwd / ".hud" / "config.json").write_text(json.dumps(ids))


def uploads(services: FakeServices) -> list[dict[str, Any]]:
    return services.bodies("api", "POST", "/v2/tasks/upload")


def test_a_dry_run_plans_creating_every_task_and_changes_nothing(
    platform: FakeServices, hud: Hud
) -> None:
    result = hud("sync", "tasks", "demo", "tasks.py", "--dry-run", "--json")

    assert (result.exit_code, result.json["create_count"], result.json["taskset"]) == (
        0,
        2,
        "demo",
    ), result
    assert (uploads(platform), linked(hud)) == ([], {})


def test_the_first_sync_creates_the_taskset_with_every_task_and_links_it(
    platform: FakeServices, hud: Hud
) -> None:
    result = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")

    assert (result.exit_code, result.json["taskset_id"]) == (0, TASKSET_ID), result
    (upload,) = uploads(platform)
    assert (upload["taskset_name"], "taskset_id" in upload) == ("demo", False)
    assert [(t["name"], t["env"], t["task_id"], t["args"]) for t in upload["tasks"]] == [
        (ONE, {"name": "example"}, "solve", {"n": 1}),
        (TWO, {"name": "example"}, "solve", {"n": 2}),
    ]
    assert linked(hud)["taskset_id"] == TASKSET_ID


@pytest.mark.parametrize(
    ("remote", "uploaded", "status"),
    [((1, 2), [], "up_to_date"), ((1,), [TWO], "synced")],
    ids=["an-unchanged-taskset-uploads-nothing", "a-new-task-uploads-only-itself"],
)
def test_a_sync_uploads_only_what_the_remote_taskset_lacks(
    remote: tuple[int, ...], uploaded: list[str], status: str, platform: FakeServices, hud: Hud
) -> None:
    holding(platform, TASKSET_ID, "demo", exported(*remote))

    result = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")

    assert (result.exit_code, result.json["status"]) == (0, status), result
    assert [[t["name"] for t in u["tasks"]] for u in uploads(platform)] == (
        [uploaded] if uploaded else []
    )
    assert all(u["taskset_id"] == TASKSET_ID for u in uploads(platform))


def test_a_bare_sync_uses_the_linked_taskset_by_its_remote_name(
    platform: FakeServices, hud: Hud
) -> None:
    holding(platform, TASKSET_ID, "renamed", exported(1, 2))
    link(hud, taskset_id=TASKSET_ID)

    result = hud("sync", "--json")

    assert (result.exit_code, result.json["status"], result.json["taskset"]) == (
        0,
        "up_to_date",
        "renamed",
    ), result


@pytest.mark.parametrize(
    ("options", "uploaded"),
    [([], [ONE, TWO]), (["--task", ONE], [ONE])],
    ids=["every-task", "one-task-by-slug"],
)
def test_force_uploads_regardless_of_what_the_remote_holds(
    options: list[str], uploaded: list[str], platform: FakeServices, hud: Hud
) -> None:
    holding(platform, TASKSET_ID, "renamed", exported(1, 2))
    link(hud, taskset_id=TASKSET_ID)

    result = hud("sync", "tasks", "--force", *options, "--yes", "--json")

    assert result.exit_code == 0, result
    (upload,) = uploads(platform)
    assert (upload["taskset_id"], upload["taskset_name"]) == (TASKSET_ID, "renamed")
    assert [task["name"] for task in upload["tasks"]] == uploaded


@pytest.mark.parametrize(
    ("options", "relinked"), [([], False), (["--link"], True)], ids=["kept", "relinked"]
)
def test_syncing_another_taskset_moves_the_link_only_when_asked(
    options: list[str], relinked: bool, platform: FakeServices, hud: Hud
) -> None:
    holding(platform, OTHER_ID, "other", [])
    platform.route(
        "api",
        "POST",
        "/v2/tasks/upload",
        json={"taskset_id": OTHER_ID, "tasks_created": 2, "tasks_updated": 0},
    )
    link(hud, taskset_id=TASKSET_ID)

    result = hud("sync", "tasks", "other", "tasks.py", *options, "--yes", "--json")

    assert result.exit_code == 0, result
    assert linked(hud)["taskset_id"] == (OTHER_ID if relinked else TASKSET_ID)


def test_project_places_the_upload_without_linking_it(platform: FakeServices, hud: Hud) -> None:
    project = {"id": PROJECT_ID, "name": "browser-evals", "capabilities": {"create": True}}
    platform.route("api", "GET", f"/v2/projects/{PROJECT_ID}", json=project)

    result = hud("sync", "tasks", "demo", "tasks.py", "--project", PROJECT_ID, "--yes", "--json")

    assert result.exit_code == 0, result
    assert [upload["project_id"] for upload in uploads(platform)] == [PROJECT_ID]
    assert linked(hud) == {}


def test_a_read_only_project_is_refused_before_anything_uploads(
    platform: FakeServices, hud: Hud
) -> None:
    locked = {"id": LOCKED_ID, "name": "locked-down", "capabilities": {}}
    platform.route("api", "GET", f"/v2/projects/{LOCKED_ID}", json=locked)

    result = hud("sync", "tasks", "demo", "tasks.py", "--project", LOCKED_ID, "--yes", "--json")

    assert (result.exit_code, result.json["error"], uploads(platform)) == (
        1,
        "permission_denied",
        [],
    ), result


def test_a_linked_taskset_that_was_deleted_is_not_found(platform: FakeServices, hud: Hud) -> None:
    link(hud, taskset_id=TASKSET_ID)
    platform.route("api", "GET", f"/v2/tasksets/{TASKSET_ID}", status=404, json={"detail": "No"})

    result = hud("sync", "tasks", "--force", "--yes", "--json")

    assert (result.exit_code, result.json["error"], uploads(platform)) == (1, "not_found", [])


def test_nothing_linked_and_no_taskset_named_is_a_usage_error(
    platform: FakeServices, hud: Hud
) -> None:
    result = hud("sync", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert result.json["message"].startswith("No taskset specified.")


@pytest.mark.parametrize("target", ["tasks.json", "out/tasks.jsonl"])
def test_an_exported_taskset_reads_back_as_the_same_tasks(
    target: str, platform: FakeServices, hud: Hud
) -> None:
    remote = [
        {"name": "one", "env": "e", "scenario": "solve", "args": {"n": 1}},
        {"name": "two", "env": "e", "scenario": "solve", "args": {"n": {"x": 2}}},
    ]
    holding(platform, TASKSET_ID, "demo", remote)

    result = hud("sync", "tasks", "demo", "--export", target, "--json")
    rows = hud("task", "list", "--source", target, "--json")

    assert result.exit_code == 0, result
    assert [(r["slug"], r["id"], r["args"]) for r in rows.json] == [
        (t["name"], t["scenario"], t["args"]) for t in remote
    ]


def test_a_csv_export_has_a_column_per_argument(platform: FakeServices, hud: Hud) -> None:
    remote = [
        {"name": "one", "env": "e", "scenario": "solve", "args": {"n": 1}},
        {"name": "two", "env": "e", "scenario": "solve", "args": {"n": {"x": 2}}},
    ]
    holding(platform, TASKSET_ID, "demo", remote)

    result = hud("sync", "tasks", "demo", "--export", "tasks.csv", "--json")

    assert result.exit_code == 0, result
    with (hud.cwd / "tasks.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["slug"], r["env"], r["id"], json.loads(r["arg:n"])) for r in rows] == [
        ("one", "e", "solve", 1),
        ("two", "e", "solve", {"x": 2}),
    ]


def test_tasks_for_another_environment_than_the_linked_one_are_flagged(
    platform: FakeServices, hud: Hud
) -> None:
    platform.route("api", "GET", f"/v2/registry/{ENV_ID}", json={"id": ENV_ID, "name": "browser"})
    link(hud, registry_id=ENV_ID)

    result = hud("sync", "tasks", "demo", "tasks.py", "--dry-run")

    assert result.exit_code == 0, result
    assert "'browser': example" in " ".join(result.stderr.split())


@pytest.mark.parametrize(
    "ref", ["browser_anchor", ENV_ID], ids=["by-name-across-separators", "by-id"]
)
def test_sync_env_links_the_directory_to_the_environment_it_names(
    ref: str, signed_in: FakeServices, hud: Hud
) -> None:
    envs = [
        {"id": "bbbbbbbb-0000-4000-8000-000000000001", "name": "browser"},
        {"id": ENV_ID, "name": "browser-anchor"},
    ]
    signed_in.route("api", "GET", "/v2/registry", json={"items": envs, "total": 2})
    signed_in.route("api", "GET", f"/v2/registry/{ENV_ID}", json=envs[1])

    first = hud("sync", "env", ref, "--json")
    again = hud("sync", "env", ref, "--json")

    assert (first.exit_code, first.json["changed"], again.json["changed"]) == (0, True, False)
    assert linked(hud)["registry_id"] == ENV_ID


def test_sync_env_for_an_environment_that_is_gone_is_not_found(
    signed_in: FakeServices, hud: Hud
) -> None:
    signed_in.route("api", "GET", f"/v2/registry/{ENV_ID}", status=404, json={"detail": "No"})

    result = hud("sync", "env", ENV_ID, "--json")

    assert (result.exit_code, result.json["error"], linked(hud)) == (1, "not_found", {}), result
