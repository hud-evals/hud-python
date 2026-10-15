"""``hud sync``: push local tasks to a platform taskset, export it, and link an environment."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.harness import Reply

from .conftest import BROWSER_PROJECT_ID, LOCKED_PROJECT_ID

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, Request

DEMO_ID = "eeeeeeee-0000-4000-8000-000000000001"
OTHER_ID = "eeeeeeee-0000-4000-8000-000000000002"
REGISTRY_ID = "bbbbbbbb-0000-4000-8000-000000000001"
ENVIRONMENT = {"id": REGISTRY_ID, "name": "browser", "latest_build": {"version": 2}}
TASKS_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=1), solve(n=2)]
"""


class Tasksets:
    """The platform's tasksets: lookup by name or id, export, and upload."""

    def __init__(self, services: FakeServices) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        services.route("api", "GET", "/v2/tasksets/by-name/{name}", handler=self.by_name)
        services.route("api", "GET", "/v2/tasksets/{id}", handler=self.get)
        services.route("api", "GET", "/v2/tasksets/{id}/export", handler=self.export)
        services.route("api", "POST", "/v2/tasks/upload", handler=self.upload)

    def add(self, taskset_id: str, name: str, tasks: list[dict[str, Any]] | None = None) -> None:
        self.records[taskset_id] = {"name": name, "tasks": {t["name"]: t for t in tasks or []}}

    def tasks(self, taskset_id: str) -> list[tuple[str, str, dict[str, Any]]]:
        """The taskset's tasks as (env, template, args), sorted."""
        tasks = self.records[taskset_id]["tasks"].values()
        return sorted(((t["env"], t["scenario"], t["args"]) for t in tasks), key=repr)

    def by_name(self, request: Request) -> Reply:
        for taskset_id, record in self.records.items():
            if record["name"] == request.params["name"]:
                return Reply(json={"taskset_id": taskset_id, "name": record["name"]})
        return Reply(status=404, json={"detail": "Taskset not found"})

    def get(self, request: Request) -> Reply:
        record = self.records.get(request.params["id"])
        if record is None:
            return Reply(status=404, json={"detail": "Taskset not found"})
        return Reply(json={"id": request.params["id"], "name": record["name"]})

    def export(self, request: Request) -> Reply:
        record = self.records.get(request.params["id"])
        if record is None:
            return Reply(status=404, json={"detail": "Taskset not found"})
        return Reply(json={"name": record["name"], "tasks": list(record["tasks"].values())})

    def upload(self, request: Request) -> Reply:
        body = request.json
        taskset_id = (
            body.get("taskset_id") or f"eeeeeeee-0000-4000-8000-{len(self.records) + 1:012d}"
        )
        record = self.records.setdefault(taskset_id, {"name": body["taskset_name"], "tasks": {}})
        created = updated = 0
        for task in body["tasks"]:
            exported = {
                "name": task["name"],
                "env": task["env"]["name"],
                "scenario": task["task_id"],
                "args": task.get("args") or {},
            }
            updated += task["name"] in record["tasks"]
            created += task["name"] not in record["tasks"]
            record["tasks"][task["name"]] = exported
        return Reply(
            json={"taskset_id": taskset_id, "tasks_created": created, "tasks_updated": updated}
        )


@pytest.fixture
def tasksets(projects: FakeServices) -> Tasksets:
    return Tasksets(projects)


@pytest.fixture
def source(hud: Hud) -> Path:
    path = hud.cwd / "tasks.py"
    path.write_text(TASKS_PY)
    return path


def uploads(services: FakeServices) -> list[Any]:
    return services.bodies("api", "POST", "/v2/tasks/upload")


def link(hud: Hud, key: str = "taskset_id") -> Any:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()).get(key) if path.exists() else None


def test_a_taskset_round_trip(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    plan = hud("sync", "tasks", "demo", "tasks.py", "--dry-run", "--json")
    assert (plan.exit_code, plan.json["create_count"], uploads(projects), link(hud)) == (
        0,
        2,
        [],
        None,
    )

    first = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert first.exit_code == 0, first
    taskset_id = first.json["taskset_id"]
    assert tasksets.tasks(taskset_id) == [
        ("example", "solve", {"n": 1}),
        ("example", "solve", {"n": 2}),
    ]
    assert link(hud) == taskset_id

    again = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert (again.json["status"], len(uploads(projects))) == ("up_to_date", 1)

    source.write_text(TASKS_PY.replace("solve(n=2)]", "solve(n=2), solve(n=3)]"))
    edited = hud("sync", "tasks", "--yes", "--json")
    assert edited.exit_code == 0, edited
    assert [task["args"] for task in uploads(projects)[-1]["tasks"]] == [{"n": 3}]
    assert len(tasksets.tasks(taskset_id)) == 3

    tasksets.records[taskset_id]["name"] = "renamed"
    bare = hud("sync", "--json")
    assert (bare.exit_code, bare.json["status"], bare.json["taskset"]) == (
        0,
        "up_to_date",
        "renamed",
    )
    forced = hud("sync", "tasks", "--force", "--yes", "--json")
    assert forced.exit_code == 0, forced
    last = uploads(projects)[-1]
    assert (last["taskset_id"], last["taskset_name"], len(last["tasks"])) == (
        taskset_id,
        "renamed",
        3,
    )

    slug = uploads(projects)[0]["tasks"][0]["name"]
    one = hud("sync", "tasks", "--task", slug, "--force", "--yes", "--json")
    assert one.exit_code == 0, one
    assert [task["name"] for task in uploads(projects)[-1]["tasks"]] == [slug]

    tasksets.add(OTHER_ID, "other")
    elsewhere = hud("sync", "tasks", OTHER_ID, "tasks.py", "--force", "--yes", "--json")
    planned = hud("sync", "tasks", OTHER_ID, "tasks.py", "--link", "--dry-run", "--json")
    assert (elsewhere.exit_code, planned.exit_code, link(hud)) == (0, 0, taskset_id)
    assert len(tasksets.tasks(OTHER_ID)) == 3
    relinked = hud("sync", "tasks", OTHER_ID, "tasks.py", "--link", "--yes", "--json")
    assert (relinked.exit_code, link(hud)) == (0, OTHER_ID)


def test_a_stored_taskset_that_was_deleted_is_never_recreated(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    tasksets.records.clear()

    result = hud("sync", "tasks", "--force", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result
    assert (len(uploads(projects)), tasksets.records) == (1, {})


REMOTE: list[dict[str, Any]] = [
    {"name": "one", "env": "e", "scenario": "solve", "args": {"n": 1}},
    {"name": "two", "env": "e", "scenario": "solve", "args": {"n": {"x": 2}}},
]


@pytest.mark.parametrize("target", ["tasks.json", "out/tasks.jsonl"])
def test_an_exported_taskset_reads_back_as_the_same_tasks(
    hud: Hud, tasksets: Tasksets, target: str
) -> None:
    tasksets.add(DEMO_ID, "demo", REMOTE)

    result = hud("sync", "tasks", "demo", "--export", target, "--json")
    rows = hud("task", "list", "--source", target, "--json")

    assert (result.exit_code, result.json["task_count"]) == (0, 2), result
    assert [(r["slug"], r["id"], r["args"]) for r in rows.json] == [
        (t["name"], t["scenario"], t["args"]) for t in REMOTE
    ]


def test_a_csv_export_has_a_column_per_argument(hud: Hud, tasksets: Tasksets) -> None:
    tasksets.add(DEMO_ID, "demo", REMOTE)

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
        pytest.param(
            ["tasks", "demo", "--export", "t.json", "--link"], 2, "usage", id="export-and-link"
        ),
        pytest.param(["tasks", "demo", "tasks.py"], 2, "usage", id="unconfirmed"),
        pytest.param(["tasks"], 2, "usage", id="no-taskset"),
        pytest.param([], 2, "usage", id="nothing-linked"),
        pytest.param(["tasks", "demo", "missing.json"], 1, "not_found", id="a-missing-source"),
        pytest.param(
            ["tasks", "demo", "tasks.py", "--task", "nope", "--yes"],
            2,
            "usage",
            id="an-unknown-slug",
        ),
        pytest.param(
            ["tasks", "nope", "--export", "t.json"], 2, "usage", id="export-of-an-unknown-taskset"
        ),
    ],
)
def test_sync_refusals_upload_nothing(
    hud: Hud,
    projects: FakeServices,
    tasksets: Tasksets,
    source: Path,
    argv: list[str],
    exit_code: int,
    error: str,
) -> None:
    result = hud("sync", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result
    assert uploads(projects) == []
    assert link(hud) is None
    assert not (hud.cwd / "t.json").exists()


def test_a_rejected_upload_fails_and_links_nothing(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    projects.route("api", "POST", "/v2/tasks/upload", Reply(status=400, json={"detail": "No"}))

    result = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")

    assert result.exit_code == 1, result
    assert link(hud) is None


@pytest.mark.parametrize(
    ("argv", "remote", "exit_code", "status"),
    [
        (["--project", LOCKED_PROJECT_ID, "--yes"], "same", 0, "up_to_date"),
        (["--project", LOCKED_PROJECT_ID, "--dry-run"], "empty", 0, None),
        (["--project", LOCKED_PROJECT_ID, "--yes"], "empty", 1, "permission_denied"),
        (["--project", BROWSER_PROJECT_ID, "--yes"], "empty", 0, "synced"),
    ],
)
def test_the_project_option_places_the_upload_without_pinning_the_directory(
    hud: Hud,
    projects: FakeServices,
    tasksets: Tasksets,
    source: Path,
    argv: list[str],
    remote: str,
    exit_code: int,
    status: str | None,
) -> None:
    rows = hud("task", "list", "--source", "tasks.py", "--json").json
    synced = [
        {"name": r["slug"], "env": "example", "scenario": r["id"], "args": r["args"]} for r in rows
    ]
    tasksets.add(DEMO_ID, "demo", synced if remote == "same" else [])

    result = hud("sync", "tasks", "demo", "tasks.py", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json.get("status", result.json.get("error")) == status
    expected = [BROWSER_PROJECT_ID] if status == "synced" else []
    assert [body["project_id"] for body in uploads(projects)] == expected
    assert link(hud) is None


def test_tasks_for_another_environment_than_the_linked_one_are_flagged(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    projects.route("api", "GET", "/v2/registry", json={"items": [ENVIRONMENT], "total": 1})
    projects.route("api", "GET", f"/v2/registry/{REGISTRY_ID}", json=ENVIRONMENT)
    assert hud("sync", "env", "browser").exit_code == 0

    result = hud("sync", "tasks", "demo", "tasks.py", "--dry-run")

    assert result.exit_code == 0, result
    assert "'browser': example" in " ".join(result.stderr.split())


# ─── sync env ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "records", "exit_code", "linked"),
    [
        pytest.param(
            ["browser_anchor"],
            [ENVIRONMENT, {**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            0,
            OTHER_ID,
            id="by-name-across-separators",
        ),
        pytest.param([REGISTRY_ID], [], 0, REGISTRY_ID, id="by-id"),
        pytest.param(
            ["Browser", "--dry-run"],
            [ENVIRONMENT, {**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            0,
            None,
            id="dry-run",
        ),
        pytest.param(
            ["anchor"],
            [{**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            1,
            None,
            id="a-partial-name",
        ),
        pytest.param([OTHER_ID], [], 1, None, id="an-inaccessible-id"),
        pytest.param([], [], 2, None, id="no-name-and-no-terminal"),
    ],
)
def test_sync_env_links_an_environment_by_name_or_id(
    hud: Hud,
    projects: FakeServices,
    argv: list[str],
    records: list[dict[str, Any]],
    exit_code: int,
    linked: str | None,
) -> None:
    def search(request: Request) -> Reply:
        term = request.query["search"][0]
        matches = [record for record in records if term in record["name"]]
        return Reply(json={"items": matches, "total": len(matches)})

    projects.route("api", "GET", "/v2/registry", handler=search)
    projects.route("api", "GET", f"/v2/registry/{REGISTRY_ID}", json=ENVIRONMENT)
    projects.route("api", "GET", f"/v2/registry/{OTHER_ID}", status=404, json={"detail": "gone"})

    result = hud("sync", "env", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert link(hud, "registry_id") == linked
