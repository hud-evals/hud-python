"""The workspace link (``.hud/config.json``) and the Project a directory places work in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from tests.harness import Reply

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
    assert config(env, registry) == snapshot(
        {
            "version": 1,
            "scope": {
                "origin": "http://hud.test",
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            },
            "registry_id": "bbbbbbbb-0000-4000-8000-000000000001",
            "taskset_id": None,
            "project_id": "aaaaaaaa-0000-4000-8000-000000000002",
        }
    )
    assert (linked.json["changed"], relinked.json["changed"]) == (True, False)
    assert (env / ".hud" / "config.json").read_bytes() == written
    assert shown.json == snapshot(
        {
            "project": {
                "id": "aaaaaaaa-0000-4000-8000-000000000002",
                "name": "browser-evals",
                "is_default": False,
                "can_create": True,
            },
            "source": ".hud/config.json",
            "label": "browser-evals (via .hud/config.json)",
        }
    )
    assert "sk-hud-test" not in written.decode()


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        (
            {"origin": "http://hud.test", "user_id": OTHER_ID, "team_id": TEAM_ID},
            "was linked with different HUD credentials than the ones currently in use.",
        ),
        (
            {"origin": "http://hud.test", "user_id": USER_ID, "team_id": OTHER_ID},
            "was linked with different HUD credentials than the ones currently in use.",
        ),
        (
            {"origin": "https://api.example", "user_id": USER_ID, "team_id": TEAM_ID},
            "was linked against https://api.example, not http://hud.test.",
        ),
    ],
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
    message: str,
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
    assert result.json["message"].replace(origin, "http://hud.test").endswith(message)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("files", "argv", "exit_code", "after"),
    [
        pytest.param(
            {
                "config.json": json.dumps(
                    {
                        "registryId": REGISTRY_ID,
                        "registryName": "example",
                        "tasksetId": OTHER_ID,
                        "syncEnv": True,
                    }
                )
            },
            ["project", "use", "browser-evals"],
            0,
            snapshot(
                {
                    "config.json": """\
{
  "version": 1,
  "scope": {
    "origin": "http://hud.test",
    "user_id": "11111111-1111-4111-8111-111111111111",
    "team_id": "22222222-2222-4222-8222-222222222222"
  },
  "registry_id": "bbbbbbbb-0000-4000-8000-000000000001",
  "taskset_id": "cccccccc-0000-4000-8000-000000000001",
  "project_id": "aaaaaaaa-0000-4000-8000-000000000002"
}
"""
                }
            ),
            id="camelcase-rewritten",
        ),
        pytest.param(
            {"config.json": json.dumps({"registryId": REGISTRY_ID})},
            ["project", "use", "browser-evals", "--dry-run"],
            0,
            snapshot({"config.json": '{"registryId": "bbbbbbbb-0000-4000-8000-000000000001"}'}),
            id="camelcase-dry-run-untouched",
        ),
        pytest.param(
            {"config.json": '{"broken":'},
            ["project", "use", "browser-evals"],
            1,
            snapshot({"config.json": '{"broken":'}),
            id="corrupt-untouched",
        ),
        pytest.param(
            {"deploy.json": '{"registryId":"old","syncEnv":true}'},
            ["project", "use", "browser-evals", "--dry-run"],
            0,
            snapshot({"deploy.json": '{"registryId":"old","syncEnv":true}'}),
            id="legacy-deploy-json-ignored",
        ),
    ],
)
def test_older_and_damaged_configs(
    hud: Hud,
    registry: FakeServices,
    files: dict[str, str],
    argv: list[str],
    exit_code: int,
    after: dict[str, str],
) -> None:
    directory = hud.cwd / ".hud"
    directory.mkdir()
    for name, text in files.items():
        (directory / name).write_text(text)

    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    origin = registry.url("api").removesuffix("/api")
    assert {
        path.name: path.read_text().replace(origin, "http://hud.test")
        for path in sorted(directory.iterdir())
    } == after


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
    ("linked", "default", "exit_code", "shown"),
    [
        (
            None,
            None,
            0,
            snapshot({"project": None, "source": "team default", "label": "team default Project"}),
        ),
        (
            None,
            "Browser Evals",
            0,
            snapshot(
                {
                    "project": {
                        "id": "aaaaaaaa-0000-4000-8000-000000000002",
                        "name": "browser-evals",
                        "is_default": False,
                        "can_create": True,
                    },
                    "source": "HUD_DEFAULT_PROJECT",
                    "label": "browser-evals (via HUD_DEFAULT_PROJECT)",
                }
            ),
        ),
        (
            None,
            BROWSER_PROJECT_ID.upper(),
            0,
            snapshot(
                {
                    "project": {
                        "id": "aaaaaaaa-0000-4000-8000-000000000002",
                        "name": "browser-evals",
                        "is_default": False,
                        "can_create": True,
                    },
                    "source": "HUD_DEFAULT_PROJECT",
                    "label": "browser-evals (via HUD_DEFAULT_PROJECT)",
                }
            ),
        ),
        (
            BROWSER_PROJECT_ID,
            "locked-down",
            0,
            snapshot(
                {
                    "project": {
                        "id": "aaaaaaaa-0000-4000-8000-000000000002",
                        "name": "browser-evals",
                        "is_default": False,
                        "can_create": True,
                    },
                    "source": ".hud/config.json",
                    "label": "browser-evals (via .hud/config.json)",
                }
            ),
        ),
        (
            None,
            "browser",
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No Project named 'browser'.",
                    "input": {"project": "browser"},
                    "suggestion": "Run 'hud project list' to see visible Projects.",
                }
            ),
        ),
        (
            None,
            "nope",
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "No Project named 'nope'.",
                    "input": {"project": "nope"},
                    "suggestion": "Run 'hud project list' to see visible Projects.",
                }
            ),
        ),
        (
            None,
            OTHER_ID,
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "Request failed: Not found",
                    "input": {"project": "cccccccc-0000-4000-8000-000000000001"},
                    "suggestion": "Check the project id, or list existing ones.",
                }
            ),
        ),
    ],
)
def test_project_shows_the_selected_project_and_why(
    hud: Hud,
    projects: FakeServices,
    hud_env: HudEnv,
    linked: str | None,
    default: str | None,
    exit_code: int,
    shown: dict[str, Any],
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT=default)
    if linked is not None:
        assert hud("project", "use", linked).exit_code == 0

    result = hud("project", "--json")

    assert result.exit_code == exit_code, result
    assert result.json == shown


def test_the_team_default_needs_no_project_lookup(hud: Hud, projects: FakeServices) -> None:
    result = hud("project")

    assert result.exit_code == 0, result
    assert result.stderr == snapshot("Project: team default Project\n")
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
    assert [request.query for request in platform.requests("api", "GET", "/v2/projects")] == [
        {"limit": ["500"], "offset": ["0"]},
        {"limit": ["500"], "offset": ["50"]},
    ]


def test_project_list_marks_default_and_read_only(hud: Hud, projects: FakeServices) -> None:
    text = hud("project", "list")
    document = hud("project", "list", "--json")

    assert (text.exit_code, document.exit_code) == (0, 0)
    assert text.stderr == snapshot("""\
default  aaaaaaaa-0000-4000-8000-000000000001 (default)
browser-evals  aaaaaaaa-0000-4000-8000-000000000002
locked-down  aaaaaaaa-0000-4000-8000-000000000003 (read-only)
""")
    assert document.json == snapshot(
        [
            {
                "id": "aaaaaaaa-0000-4000-8000-000000000001",
                "name": "default",
                "is_default": True,
                "can_create": True,
            },
            {
                "id": "aaaaaaaa-0000-4000-8000-000000000002",
                "name": "browser-evals",
                "is_default": False,
                "can_create": True,
            },
            {
                "id": "aaaaaaaa-0000-4000-8000-000000000003",
                "name": "locked-down",
                "is_default": False,
                "can_create": False,
            },
        ]
    )


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

    assert result.exit_code == 0, result
    assert result.json == {
        "id": BROWSER_PROJECT_ID,
        "name": "browser-evals",
        "is_default": False,
        "can_create": True,
    }
    assert projects.bodies("api", "POST", "/v2/projects") == [
        {"name": "Browser Evals", **({"description": "UI tasks"} if linked else {})}
    ]
    stored = hud.cwd / ".hud" / "config.json"
    assert stored.exists() is linked
    if linked:
        assert json.loads(stored.read_text())["project_id"] == BROWSER_PROJECT_ID


@pytest.mark.parametrize(
    ("argv", "reply", "exit_code", "document"),
    [
        (
            ["project", "create", "Browser Evals", "--dry-run"],
            None,
            0,
            snapshot({"dry_run": True, "action": "create_project", "name": "Browser Evals"}),
        ),
        (
            ["project", "create", "Browser Evals", "--no-use"],
            Reply(status=403, json={"detail": "Projects are not enabled"}),
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: Projects are not enabled",
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
        ),
        (
            ["project", "use", "locked-down"],
            None,
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "You do not have permission to create environments or "
                    "tasksets in project 'locked-down'",
                    "input": {"project": "aaaaaaaa-0000-4000-8000-000000000003"},
                }
            ),
        ),
        (
            ["project", "use", "browser-evals", "--dry-run"],
            None,
            0,
            snapshot(
                {
                    "id": "aaaaaaaa-0000-4000-8000-000000000002",
                    "name": "browser-evals",
                    "is_default": False,
                    "can_create": True,
                    "dry_run": True,
                }
            ),
        ),
    ],
)
def test_project_changes_that_write_nothing(
    hud: Hud,
    projects: FakeServices,
    argv: list[str],
    reply: Reply | None,
    exit_code: int,
    document: dict[str, Any],
) -> None:
    if reply is not None:
        projects.route("api", "POST", "/v2/projects", reply)

    result = hud(*argv, "--json")

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert not (hud.cwd / ".hud").exists()


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
