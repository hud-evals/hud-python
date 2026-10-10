"""``hud sync``: push local tasks to a platform taskset, export it, and link an environment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from tests.harness import Reply, scrub

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


def link(hud: Hud) -> Any:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text())["taskset_id"] if path.exists() else None


def test_a_taskset_round_trip(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    plan = hud("sync", "tasks", "demo", "tasks.py", "--dry-run", "--json")
    assert (plan.exit_code, plan.json, uploads(projects), link(hud)) == (
        0,
        snapshot(
            {
                "taskset": "demo",
                "create_count": 2,
                "update_count": 0,
                "unchanged_count": 0,
                "remote_only_count": 0,
                "to_apply": ["solve", "solve"],
                "dry_run": True,
                "action": "sync_tasks",
            }
        ),
        [],
        None,
    )

    first = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert first.exit_code == 0, first
    assert first.json == snapshot(
        {
            "taskset": "demo",
            "create_count": 2,
            "update_count": 0,
            "unchanged_count": 0,
            "remote_only_count": 0,
            "to_apply": ["solve", "solve"],
            "status": "synced",
            "tasks_created": 2,
            "tasks_updated": 0,
            "taskset_id": "eeeeeeee-0000-4000-8000-000000000001",
        }
    )
    assert uploads(projects) == snapshot(
        [
            {
                "taskset_name": "demo",
                "tasks": [
                    {
                        "name": "solve-bae34777",
                        "env": {"name": "example"},
                        "task_id": "solve",
                        "args": {"n": 1},
                    },
                    {
                        "name": "solve-99dd84a6",
                        "env": {"name": "example"},
                        "task_id": "solve",
                        "args": {"n": 2},
                    },
                ],
            }
        ]
    )
    taskset_id = first.json["taskset_id"]
    assert link(hud) == taskset_id

    again = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")
    assert (again.json["status"], len(uploads(projects))) == ("up_to_date", 1)

    source.write_text(TASKS_PY.replace("solve(n=2)]", "solve(n=2), solve(n=3)]"))
    edited = hud("sync", "tasks", "--yes", "--json")
    assert edited.exit_code == 0, edited
    assert edited.json == snapshot(
        {
            "taskset": "demo",
            "create_count": 1,
            "update_count": 0,
            "unchanged_count": 2,
            "remote_only_count": 0,
            "to_apply": ["solve"],
            "status": "synced",
            "tasks_created": 1,
            "tasks_updated": 0,
            "taskset_id": "eeeeeeee-0000-4000-8000-000000000001",
        }
    )
    assert uploads(projects)[-1]["taskset_id"] == taskset_id

    tasksets.records[taskset_id]["name"] = "renamed"
    bare = hud("sync", "--json")
    assert (bare.exit_code, bare.json["status"], bare.json["taskset"]) == (
        0,
        "up_to_date",
        "renamed",
    )
    forced = hud("sync", "tasks", "--force", "--yes", "--json")
    assert forced.exit_code == 0, forced
    assert uploads(projects)[-1]["taskset_name"] == "renamed"
    assert len(uploads(projects)[-1]["tasks"]) == 3
    assert link(hud) == taskset_id

    slug = uploads(projects)[0]["tasks"][0]["name"]
    one = hud("sync", "tasks", "--task", slug, "--force", "--yes", "--json")
    assert one.exit_code == 0, one
    assert [task["name"] for task in uploads(projects)[-1]["tasks"]] == [slug]

    tasksets.add(OTHER_ID, "other")
    elsewhere = hud("sync", "tasks", OTHER_ID, "tasks.py", "--force", "--yes", "--json")
    planned = hud("sync", "tasks", OTHER_ID, "tasks.py", "--link", "--dry-run", "--json")
    assert (elsewhere.exit_code, planned.exit_code, link(hud)) == (0, 0, taskset_id)
    assert uploads(projects)[-1]["taskset_id"] == OTHER_ID
    relinked = hud("sync", "tasks", OTHER_ID, "tasks.py", "--link", "--yes", "--json")
    assert (relinked.exit_code, link(hud)) == (0, OTHER_ID)


def test_a_stored_taskset_that_was_deleted_is_never_recreated(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    tasksets.records.clear()

    result = hud("sync", "tasks", "--force", "--yes", "--json")

    assert result.exit_code == 1, result
    assert scrub(result.stdout) == snapshot("""\
{
  "error": "not_found",
  "message": "Request failed: Taskset not found",
  "suggestion": "Check the resource id, or list existing ones."
}
""")
    assert len(uploads(projects)) == 1


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            "tasks.csv",
            snapshot("""\
slug,id,env,arg:n
one,solve,e,1
two,solve,e,"{""x"": 2}"
"""),
        ),
        (
            "tasks.json",
            snapshot("""\
[
  {
    "env": "e",
    "id": "solve",
    "args": {
      "n": 1
    },
    "slug": "one"
  },
  {
    "env": "e",
    "id": "solve",
    "args": {
      "n": {
        "x": 2
      }
    },
    "slug": "two"
  }
]
"""),
        ),
        (
            "out/tasks.jsonl",
            snapshot("""\
{"env": "e", "id": "solve", "args": {"n": 1}, "slug": "one"}
{"env": "e", "id": "solve", "args": {"n": {"x": 2}}, "slug": "two"}
"""),
        ),
    ],
)
def test_export_writes_the_remote_taskset_to_a_file(
    hud: Hud, tasksets: Tasksets, target: str, expected: str
) -> None:
    tasksets.add(
        DEMO_ID,
        "demo",
        [
            {"name": "one", "env": "e", "scenario": "solve", "args": {"n": 1}},
            {"name": "two", "env": "e", "scenario": "solve", "args": {"n": {"x": 2}}},
        ],
    )

    result = hud("sync", "tasks", "demo", "--export", target, "--json")

    assert result.exit_code == 0, result
    assert result.json == {"action": "export", "taskset": "demo", "path": target, "task_count": 2}
    assert (hud.cwd / target).read_text() == expected


@pytest.mark.parametrize(
    ("argv", "exit_code", "document"),
    [
        (
            ["sync", "tasks", "demo", "--export", "t.json", "--link"],
            2,
            snapshot({"error": "usage", "message": "--link cannot be combined with --export"}),
        ),
        (
            ["sync", "tasks", "demo", "tasks.py"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Confirmation required in a non-interactive terminal.",
                    "suggestion": "Re-run with --yes to continue.",
                }
            ),
        ),
        (
            ["sync", "tasks"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "No taskset specified. Pass a taskset name/ID or run 'hud sync tasks "
                        "<name>' first to store it."
                    ),
                }
            ),
        ),
        (
            ["sync"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "No taskset specified. Pass a taskset name/ID or run 'hud sync tasks "
                        "<name>' first to store it."
                    ),
                }
            ),
        ),
        (
            ["sync", "tasks", "demo", "missing.json"],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "[Errno 2] No such file or directory: 'missing.json'",
                }
            ),
        ),
        (
            ["sync", "tasks", "demo", "tasks.py", "--task", "nope", "--yes"],
            2,
            snapshot({"error": "usage", "message": "No task found with slug 'nope'"}),
        ),
        (
            ["sync", "tasks", "nope", "--export", "t.json"],
            2,
            snapshot({"error": "usage", "message": "taskset not found: nope"}),
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
    document: dict[str, Any],
) -> None:
    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    assert json.loads(scrub(result.stdout, hud.cwd)) == document
    assert uploads(projects) == []
    assert link(hud) is None


@pytest.mark.parametrize(
    ("reply", "document"),
    [
        (
            Reply(status=400, json={"detail": "Taskset belongs to another Project"}),
            snapshot(
                {
                    "error": "failure",
                    "message": "Request failed: Taskset belongs to another Project",
                    "input": {"taskset": "demo"},
                }
            ),
        ),
        (
            Reply(status=403, json={"detail": "Upload rejected"}),
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: Upload rejected",
                    "input": {"taskset": "demo"},
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
        ),
        (
            Reply(status=500, json={"detail": "Upload rejected"}),
            snapshot(
                {
                    "error": "server_error",
                    "message": "Request failed: Upload rejected",
                    "input": {"taskset": "demo"},
                    "suggestion": "Retry; this error is often transient.",
                }
            ),
        ),
    ],
)
def test_a_rejected_upload_fails_and_links_nothing(
    hud: Hud,
    projects: FakeServices,
    tasksets: Tasksets,
    source: Path,
    reply: Reply,
    document: dict[str, Any],
) -> None:
    projects.route("api", "POST", "/v2/tasks/upload", reply)

    result = hud("sync", "tasks", "demo", "tasks.py", "--yes", "--json")

    assert result.exit_code == 1, result
    assert result.json == document
    assert "Sync complete" not in result.stderr
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
    rows = json.loads(hud("task", "list", "--source", "tasks.py", "--json").stdout)
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


def test_a_mismatched_linked_environment_warns(
    hud: Hud, projects: FakeServices, tasksets: Tasksets, source: Path
) -> None:
    projects.route("api", "GET", "/v2/registry", json={"items": [ENVIRONMENT], "total": 1})
    projects.route("api", "GET", f"/v2/registry/{REGISTRY_ID}", json=ENVIRONMENT)
    assert hud("sync", "env", "browser").exit_code == 0

    result = hud("sync", "tasks", "demo", "tasks.py", "--dry-run")

    assert result.exit_code == 0, result
    assert (
        "Local task env names do not match the linked platform environment 'browser': example"
        in " ".join(result.stderr.split())
    )


# ─── sync env ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "records", "exit_code", "document"),
    [
        (
            ["browser_anchor"],
            [ENVIRONMENT, {**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            0,
            snapshot(
                {
                    "name": "browser-anchor",
                    "id": "eeeeeeee-0000-4000-8000-000000000002",
                    "short_id": "eeeeeeee",
                    "changed": True,
                }
            ),
        ),
        (
            ["Browser", "--dry-run"],
            [ENVIRONMENT, {**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            0,
            snapshot(
                {
                    "dry_run": True,
                    "action": "link_environment",
                    "id": "bbbbbbbb-0000-4000-8000-000000000001",
                    "name": "browser",
                }
            ),
        ),
        (
            ["anchor"],
            [{**ENVIRONMENT, "id": OTHER_ID, "name": "browser-anchor"}],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No environment named 'anchor'.",
                    "input": {"environment": "anchor"},
                    "suggestion": "Run 'hud sync env' to pick from your environments.",
                }
            ),
        ),
        (
            [REGISTRY_ID],
            [],
            0,
            snapshot(
                {
                    "name": "browser",
                    "id": "bbbbbbbb-0000-4000-8000-000000000001",
                    "short_id": "bbbbbbbb",
                    "changed": True,
                }
            ),
        ),
        (
            [OTHER_ID],
            [],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": (
                        "Environment eeeeeeee-0000-4000-8000-000000000002 is inaccessible or "
                        "deleted."
                    ),
                    "suggestion": (
                        "Run 'hud sync env <name-or-id>' to link an accessible environment."
                    ),
                }
            ),
        ),
        (
            [],
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Pass an environment name or ID for a dry run or noninteractive link."
                    ),
                }
            ),
        ),
    ],
)
def test_sync_env_links_an_environment_by_name_or_id(
    hud: Hud,
    projects: FakeServices,
    argv: list[str],
    records: list[dict[str, Any]],
    exit_code: int,
    document: dict[str, Any],
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
    assert result.json == document
