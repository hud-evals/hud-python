"""The workspace link (``.hud/config.json``) and the Project a directory places work in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY

import pytest

from tests.harness import Reply, write_files

from .conftest import (
    BROWSER_PROJECT_ID,
    DEFAULT_PROJECT_ID,
    TEAM_ID,
    USER_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, HudEnv

REGISTRY_ID = "bbbbbbbb-0000-4000-8000-000000000001"
OTHER_ID = "cccccccc-0000-4000-8000-000000000001"
PROJECT_CREATED = {
    "id": BROWSER_PROJECT_ID,
    "name": "browser-evals",
    "capabilities": {"create": True},
}


@pytest.fixture
def registry(projects: FakeServices) -> FakeServices:
    """One deployed environment, ``example``."""
    record = {"id": REGISTRY_ID, "name": "example", "latest_build": {"version": 3}}
    projects.route("api", "GET", f"/v2/registry/{REGISTRY_ID}", json=record)
    projects.route("api", "GET", "/v2/registry", json={"items": [record], "total": 1})
    return projects


def config(directory: Path, services: FakeServices) -> Any:
    """``.hud/config.json`` with the fake platform's origin made stable."""
    text = (directory / ".hud" / "config.json").read_text()
    origin = services.url("api").removesuffix("/api")
    return json.loads(text.replace(origin, "http://hud.test"))


def link_project(hud: Hud) -> str | None:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text())["project_id"] if path.exists() else None


def test_linking_a_directory_accumulates_its_platform_ids(
    hud: Hud, registry: FakeServices, tmp_path: Path
) -> None:
    env = tmp_path / "env"

    used = hud("project", "use", "Browser Evals", "-C", str(env), "--json")
    linked = hud("sync", "env", "example", str(env), "--json")
    written = (env / ".hud" / "config.json").read_bytes()
    relinked = hud("sync", "env", REGISTRY_ID, str(env), "--json")
    shown = hud("project", "-C", str(env), "--json")

    assert [used.exit_code, linked.exit_code, relinked.exit_code, shown.exit_code] == [0] * 4
    stored = config(env, registry)
    assert (stored["registry_id"], stored["project_id"], stored["taskset_id"]) == (
        REGISTRY_ID,
        BROWSER_PROJECT_ID,
        None,
    )
    assert (stored["scope"]["user_id"], stored["scope"]["team_id"]) == (USER_ID, TEAM_ID)
    assert (linked.json["changed"], relinked.json["changed"]) == (True, False)
    assert (env / ".hud" / "config.json").read_bytes() == written
    assert (shown.json["project"]["id"], shown.json["source"]) == (
        BROWSER_PROJECT_ID,
        ".hud/config.json",
    )
    assert "sk-hud-test" not in written.decode()


@pytest.mark.parametrize(
    "scope",
    [
        {"origin": "http://hud.test", "user_id": OTHER_ID, "team_id": TEAM_ID},
        {"origin": "http://hud.test", "user_id": USER_ID, "team_id": OTHER_ID},
        {"origin": "https://api.example", "user_id": USER_ID, "team_id": TEAM_ID},
    ],
    ids=["another-user", "another-team", "another-platform"],
)
@pytest.mark.parametrize(
    "argv",
    [
        ["project", "use", "browser-evals", "--json"],
        ["project", "--json"],
        ["sync", "env", "example", "--json"],
    ],
)
def test_a_link_made_with_other_credentials_is_refused(
    hud: Hud,
    registry: FakeServices,
    argv: list[str],
    scope: dict[str, str],
) -> None:
    origin = registry.url("api").removesuffix("/api")
    path = hud.cwd / ".hud" / "config.json"
    path.parent.mkdir()
    stored = {"version": 1, "scope": scope, "project_id": DEFAULT_PROJECT_ID}
    path.write_text(json.dumps(stored).replace("http://hud.test", origin))
    before = path.read_bytes()

    result = hud(*argv)

    assert result.exit_code == 1, result
    assert result.json["error"] == "failure"
    assert path.read_bytes() == before


def test_an_older_camelcase_link_is_rewritten_with_its_ids(
    hud: Hud, registry: FakeServices
) -> None:
    write_files(
        hud.cwd / ".hud",
        {"config.json": json.dumps({"registryId": REGISTRY_ID, "tasksetId": OTHER_ID})},
    )

    result = hud("project", "use", "browser-evals", "--json")

    assert result.exit_code == 0, result
    stored = config(hud.cwd, registry)
    assert (stored["registry_id"], stored["taskset_id"], stored["project_id"]) == (
        REGISTRY_ID,
        OTHER_ID,
        BROWSER_PROJECT_ID,
    )


@pytest.mark.parametrize(
    ("files", "argv", "exit_code"),
    [
        pytest.param(
            {"config.json": json.dumps({"registryId": REGISTRY_ID})},
            ["--dry-run"],
            0,
            id="a-dry-run-over-a-camelcase-link",
        ),
        pytest.param({"config.json": '{"broken":'}, [], 1, id="a-corrupt-link"),
        pytest.param(
            {"deploy.json": '{"registryId":"old"}'}, ["--dry-run"], 0, id="a-legacy-deploy-json"
        ),
    ],
)
def test_a_link_it_cannot_or_must_not_rewrite_is_left_alone(
    hud: Hud, registry: FakeServices, files: dict[str, str], argv: list[str], exit_code: int
) -> None:
    write_files(hud.cwd / ".hud", files)

    result = hud("project", "use", "browser-evals", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert {path.name: path.read_text() for path in (hud.cwd / ".hud").iterdir()} == files


def test_the_link_records_a_normalized_origin(
    hud: Hud, registry: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_URL=f"HTTP://LOCALHOST:{registry.port}/api/")

    result = hud("project", "use", "browser-evals", "--json")

    assert result.exit_code == 0, result
    stored = json.loads((hud.cwd / ".hud" / "config.json").read_text())
    assert stored["scope"]["origin"] == f"http://localhost:{registry.port}"


# ─── which Project a directory uses ─────────────────────────────────────


@pytest.mark.parametrize(
    ("linked", "default", "project", "source"),
    [
        pytest.param(None, None, None, "team default", id="team-default"),
        pytest.param(
            None, "Browser Evals", BROWSER_PROJECT_ID, "HUD_DEFAULT_PROJECT", id="default-by-name"
        ),
        pytest.param(
            None,
            BROWSER_PROJECT_ID.upper(),
            BROWSER_PROJECT_ID,
            "HUD_DEFAULT_PROJECT",
            id="default-by-id",
        ),
        pytest.param(
            BROWSER_PROJECT_ID,
            "locked-down",
            BROWSER_PROJECT_ID,
            ".hud/config.json",
            id="the-link-beats-the-default",
        ),
    ],
)
def test_project_shows_the_selected_project_and_why(
    hud: Hud,
    projects: FakeServices,
    hud_env: HudEnv,
    linked: str | None,
    default: str | None,
    project: str | None,
    source: str,
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT=default)
    if linked is not None:
        assert hud("project", "use", linked).exit_code == 0

    result = hud("project", "--json")

    assert result.exit_code == 0, result
    assert ((result.json["project"] or {}).get("id"), result.json["source"]) == (project, source)


@pytest.mark.parametrize("default", ["browser", "nope", OTHER_ID])
def test_a_default_project_that_does_not_exist_is_not_found(
    hud: Hud, projects: FakeServices, hud_env: HudEnv, default: str
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT=default)

    result = hud("project", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result


def test_the_team_default_needs_no_project_lookup(hud: Hud, projects: FakeServices) -> None:
    result = hud("project")

    assert result.exit_code == 0, result
    assert [request.query for request in projects.requests("api", "GET", "/v2/projects")] == [
        {"limit": ["1"]}
    ]
    assert projects.requests("api", "GET", "/v2/projects/{id}") == []


def test_project_list_reads_every_page(hud: Hud, platform: FakeServices) -> None:
    many = [
        {"id": f"dddddddd-0000-4000-8000-{index:012d}", "name": f"p{index}"} for index in range(51)
    ]
    platform.route(
        "api",
        "GET",
        "/v2/projects",
        Reply(json={"items": many[:50], "total": 51}),
        Reply(json={"items": many[50:], "total": 51}),
    )

    quiet = hud("project", "list", "--quiet")

    assert quiet.exit_code == 0, quiet
    assert quiet.stdout.split() == [project["id"] for project in many]


def test_project_list_marks_the_default_and_read_only_projects(
    hud: Hud, projects: FakeServices
) -> None:
    result = hud("project", "list", "--json")

    assert result.exit_code == 0, result
    assert [(p["name"], p["is_default"], p["can_create"]) for p in result.json] == [
        ("default", True, True),
        ("browser-evals", False, True),
        ("locked-down", False, False),
    ]


@pytest.mark.parametrize(
    ("argv", "linked"),
    [
        (["project", "create", "Browser Evals", "--description", "UI tasks"], True),
        (["project", "create", "Browser Evals", "--no-use"], False),
    ],
)
def test_project_create_posts_the_project_and_links_unless_told_not_to(
    hud: Hud, projects: FakeServices, argv: list[str], linked: bool
) -> None:
    projects.route("api", "POST", "/v2/projects", json=PROJECT_CREATED)

    result = hud(*argv, "--json")

    assert (result.exit_code, result.json["id"]) == (0, BROWSER_PROJECT_ID), result
    assert projects.bodies("api", "POST", "/v2/projects") == [
        {"name": "Browser Evals", **({"description": "UI tasks"} if linked else {})}
    ]
    assert link_project(hud) == (BROWSER_PROJECT_ID if linked else None)


@pytest.mark.parametrize(
    ("argv", "reply", "exit_code"),
    [
        pytest.param(["create", "Browser Evals", "--dry-run"], None, 0, id="create-dry-run"),
        pytest.param(
            ["create", "Browser Evals", "--no-use"],
            Reply(status=403, json={"detail": "Projects are not enabled"}),
            1,
            id="create-refused",
        ),
        pytest.param(["use", "locked-down"], None, 1, id="use-a-read-only-project"),
        pytest.param(["use", "browser-evals", "--dry-run"], None, 0, id="use-dry-run"),
    ],
)
def test_project_changes_that_link_nothing(
    hud: Hud, projects: FakeServices, argv: list[str], reply: Reply | None, exit_code: int
) -> None:
    if reply is not None:
        projects.route("api", "POST", "/v2/projects", reply)

    result = hud("project", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert not (hud.cwd / ".hud").exists()
    assert projects.requests("api", "POST", "/v2/projects") == ([] if reply is None else [ANY])


@pytest.mark.parametrize(
    ("before", "after", "target"),
    [
        (["-C", "group"], [], "group"),
        (["-C", "group"], ["-C", "override"], "override"),
        ([], ["--directory", "override"], "override"),
    ],
)
def test_project_directory_option_applies_before_or_after_the_verb(
    hud: Hud, projects: FakeServices, before: list[str], after: list[str], target: str
) -> None:
    result = hud("project", *before, "use", BROWSER_PROJECT_ID, *after)

    assert result.exit_code == 0, result
    linked = [path.parent.parent.name for path in hud.cwd.glob("*/.hud/config.json")]
    assert linked == [target]
