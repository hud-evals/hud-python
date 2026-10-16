"""``hud sync``: push local tasks to a platform taskset, export it, and link an environment."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply
from tests.support.platform import BROWSER_PROJECT_ID, LOCKED_PROJECT_ID

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, Platform

TASKS_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=1), solve(n=2)]
"""


@pytest.fixture
def source(hud: Hud) -> Path:
    path = hud.cwd / "tasks.py"
    path.write_text(TASKS_PY)
    return path


def link(hud: Hud, key: str = "taskset_id") -> Any:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()).get(key) if path.exists() else None


def stored(platform: Platform, taskset_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    """A taskset's tasks as (env, template, args), sorted."""
    tasks = platform.tasksets[taskset_id]["tasks"].values()
    return sorted(((t["env"], t["scenario"], t["args"]) for t in tasks), key=repr)


def test_a_taskset_round_trip(hud: Hud, platform: Platform, source: Path) -> None:
    plan = hud("sync", "tasks", "demo", "tasks.py", "--dry-run", "--json")
    assert (plan.exit_code, plan.json["create_count"], platform.tasksets, link(hud)) == (
        0,
        2,
        {},
        None,
    )

    first = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert first.exit_code == 0, first
    taskset_id = first.json["taskset_id"]
    assert stored(platform, taskset_id) == [
        ("example", "solve", {"n": 1}),
        ("example", "solve", {"n": 2}),
    ]
    assert link(hud) == taskset_id

    again = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert (again.json["status"], len(platform.uploads)) == ("up_to_date", 1)

    source.write_text(TASKS_PY.replace("solve(n=2)]", "solve(n=2), solve(n=3)]"))
    edited = hud("sync", "tasks", "--yes", "--json")
    assert edited.exit_code == 0, edited
    assert [task["args"] for task in platform.uploads[-1]["tasks"]] == [{"n": 3}]
    assert len(stored(platform, taskset_id)) == 3

    platform.tasksets[taskset_id]["name"] = "renamed"
    bare = hud("sync", "--json")
    assert (bare.exit_code, bare.json["status"], bare.json["taskset"]) == (
        0,
        "up_to_date",
        "renamed",
    )
    forced = hud("sync", "tasks", "--force", "--yes", "--json")
    assert forced.exit_code == 0, forced
    last = platform.uploads[-1]
    assert (last["taskset_id"], last["taskset_name"], len(last["tasks"])) == (
        taskset_id,
        "renamed",
        3,
    )

    slug = platform.uploads[0]["tasks"][0]["name"]
    one = hud("sync", "tasks", "--task", slug, "--force", "--yes", "--json")
    assert one.exit_code == 0, one
    assert [task["name"] for task in platform.uploads[-1]["tasks"]] == [slug]

    other = platform.add_taskset("other")
    elsewhere = hud("sync", "tasks", other, "tasks.py", "--force", "--yes", "--json")
    planned = hud("sync", "tasks", other, "tasks.py", "--link", "--dry-run", "--json")
    assert (elsewhere.exit_code, planned.exit_code, link(hud)) == (0, 0, taskset_id)
    assert len(stored(platform, other)) == 3
    relinked = hud("sync", "tasks", other, "tasks.py", "--link", "--yes", "--json")
    assert (relinked.exit_code, link(hud)) == (0, other)


def test_a_stored_taskset_that_was_deleted_is_never_recreated(
    hud: Hud, platform: Platform, source: Path
) -> None:
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    platform.tasksets.clear()

    result = hud("sync", "tasks", "--force", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result
    assert (len(platform.uploads), platform.tasksets) == (1, {})


REMOTE: list[dict[str, Any]] = [
    {"name": "one", "env": "e", "scenario": "solve", "args": {"n": 1}},
    {"name": "two", "env": "e", "scenario": "solve", "args": {"n": {"x": 2}}},
]


@pytest.mark.parametrize("target", ["tasks.json", "out/tasks.jsonl"])
def test_an_exported_taskset_reads_back_as_the_same_tasks(
    hud: Hud, platform: Platform, target: str
) -> None:
    platform.add_taskset("demo", REMOTE)

    result = hud("sync", "tasks", "demo", "--export", target, "--json")
    rows = hud("task", "list", "--source", target, "--json")

    assert (result.exit_code, result.json["task_count"]) == (0, 2), result
    assert [(r["slug"], r["id"], r["args"]) for r in rows.json] == [
        (t["name"], t["scenario"], t["args"]) for t in REMOTE
    ]


def test_a_csv_export_has_a_column_per_argument(hud: Hud, platform: Platform) -> None:
    platform.add_taskset("demo", REMOTE)

    result = hud("sync", "tasks", "demo", "--export", "tasks.csv", "--json")

    assert result.exit_code == 0, result
    with (hud.cwd / "tasks.csv").open(newline="") as exported:
        rows = list(csv.DictReader(exported))
    assert [(r["slug"], r["env"], r["id"], json.loads(r["arg:n"])) for r in rows] == [
        (t["name"], t["env"], t["scenario"], t["args"]["n"]) for t in REMOTE
    ]


@pytest.mark.parametrize(
    ("argv", "exit_code", "error"),
    [
        pytest.param(["tasks", "demo", "tasks.py"], 2, "usage", id="unconfirmed"),
        pytest.param([], 2, "usage", id="nothing-linked"),
        pytest.param(["tasks", "demo", "missing.json"], 1, "not_found", id="a-missing-source"),
    ],
)
def test_sync_refusals_upload_nothing(
    hud: Hud, platform: Platform, source: Path, argv: list[str], exit_code: int, error: str
) -> None:
    result = hud("sync", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result
    assert (platform.uploads, link(hud)) == ([], None)


def test_a_rejected_upload_fails_and_links_nothing(
    hud: Hud, platform: Platform, source: Path
) -> None:
    platform.services.route(
        "api", "POST", "/v2/tasks/upload", Reply(status=400, json={"detail": "No"})
    )

    result = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")

    assert result.exit_code == 1, result
    assert link(hud) is None


@pytest.mark.parametrize(
    ("project", "exit_code", "placed"),
    [(LOCKED_PROJECT_ID, 1, None), (BROWSER_PROJECT_ID, 0, BROWSER_PROJECT_ID)],
    ids=["a-read-only-project", "a-writable-project"],
)
def test_the_project_option_places_the_upload_without_pinning_the_directory(
    hud: Hud, platform: Platform, source: Path, project: str, exit_code: int, placed: str | None
) -> None:
    platform.add_taskset("demo")

    result = hud("sync", "tasks", "demo", "tasks.py", "--project", project, "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert [upload["project_id"] for upload in platform.uploads] == ([placed] if placed else [])
    assert link(hud) is None


def test_tasks_for_another_environment_than_the_linked_one_are_flagged(
    hud: Hud, platform: Platform, source: Path
) -> None:
    platform.add_environment("browser")
    assert hud("sync", "env", "browser").exit_code == 0

    result = hud("sync", "tasks", "demo", "tasks.py", "--dry-run")

    assert result.exit_code == 0, result
    assert "'browser': example" in " ".join(result.stderr.split())


# ─── sync env ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reference", "linked"),
    [
        pytest.param("browser_anchor", "browser-anchor", id="by-name-across-separators"),
        pytest.param("id", "browser", id="by-id"),
    ],
)
def test_sync_env_links_an_environment_by_name_or_id(
    hud: Hud, platform: Platform, reference: str, linked: str
) -> None:
    browser = platform.add_environment("browser")
    anchor = platform.add_environment("browser-anchor")
    ids = {"browser": browser, "browser-anchor": anchor}

    result = hud("sync", "env", browser if reference == "id" else reference, "--json")

    assert result.exit_code == 0, result
    assert link(hud, "registry_id") == ids[linked]


def test_sync_env_refuses_an_environment_it_cannot_see(hud: Hud, platform: Platform) -> None:
    result = hud("sync", "env", "bbbbbbbb-0000-4000-8000-999999999999", "--json")

    assert result.exit_code == 1, result
    assert link(hud, "registry_id") is None
