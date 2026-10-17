"""The workspace link (``.hud/config.json``) and the Project a directory places work in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply, write_files
from tests.support.platform import BROWSER_PROJECT_ID, DEFAULT_PROJECT_ID, TEAM_ID, USER_ID

if TYPE_CHECKING:
    from tests.support import Hud, HudEnv, Platform

OTHER_ID = "cccccccc-0000-4000-8000-000000000001"


def link(hud: Hud) -> dict[str, Any] | None:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_linking_a_directory_accumulates_its_platform_ids(hud: Hud, platform: Platform) -> None:
    example = platform.add_environment("example")

    used = hud("project", "use", "Browser Evals", "--json")
    linked = hud("sync", "env", "example", "--json")
    written = (hud.cwd / ".hud" / "config.json").read_bytes()
    relinked = hud("sync", "env", example, "--json")
    shown = hud("project", "--json")

    assert [used.exit_code, linked.exit_code, relinked.exit_code, shown.exit_code] == [0] * 4
    stored = link(hud)
    assert stored is not None
    assert (stored["registry_id"], stored["project_id"], stored["taskset_id"]) == (
        example,
        BROWSER_PROJECT_ID,
        None,
    )
    assert (stored["scope"]["user_id"], stored["scope"]["team_id"]) == (USER_ID, TEAM_ID)
    assert (linked.json["changed"], relinked.json["changed"]) == (True, False)
    assert (hud.cwd / ".hud" / "config.json").read_bytes() == written
    assert (shown.json["project"]["id"], shown.json["source"]) == (
        BROWSER_PROJECT_ID,
        ".hud/config.json",
    )
    assert "sk-hud-test" not in written.decode()


@pytest.mark.parametrize(
    "scope",
    [
        {"origin": "here", "user_id": OTHER_ID, "team_id": TEAM_ID},
        {"origin": "here", "user_id": USER_ID, "team_id": OTHER_ID},
        {"origin": "https://api.example", "user_id": USER_ID, "team_id": TEAM_ID},
    ],
    ids=["another-user", "another-team", "another-platform"],
)
def test_a_link_made_with_other_credentials_is_refused(
    hud: Hud, platform: Platform, scope: dict[str, str]
) -> None:
    origin = platform.services.url("api").removesuffix("/api")
    stored = {"version": 1, "scope": scope, "project_id": DEFAULT_PROJECT_ID}
    write_files(
        hud.cwd / ".hud", {"config.json": json.dumps(stored).replace('"here"', f'"{origin}"')}
    )
    before = (hud.cwd / ".hud" / "config.json").read_bytes()

    result = hud("project", "use", "browser-evals", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "failure"), result
    assert (hud.cwd / ".hud" / "config.json").read_bytes() == before


def test_an_older_camelcase_link_is_rewritten_with_its_ids(hud: Hud, platform: Platform) -> None:
    example = platform.add_environment("example")
    write_files(
        hud.cwd / ".hud",
        {"config.json": json.dumps({"registryId": example, "tasksetId": OTHER_ID})},
    )

    result = hud("project", "use", "browser-evals", "--json")

    assert result.exit_code == 0, result
    stored = link(hud)
    assert stored is not None
    assert (stored["registry_id"], stored["taskset_id"], stored["project_id"]) == (
        example,
        OTHER_ID,
        BROWSER_PROJECT_ID,
    )


@pytest.mark.parametrize(
    ("files", "argv", "exit_code"),
    [
        pytest.param(
            {"config.json": json.dumps({"registryId": OTHER_ID})},
            ["--dry-run"],
            0,
            id="a-dry-run-over-a-camelcase-link",
        ),
        pytest.param({"config.json": '{"broken":'}, [], 1, id="a-corrupt-link"),
    ],
)
def test_a_link_it_cannot_or_must_not_rewrite_is_left_alone(
    hud: Hud, platform: Platform, files: dict[str, str], argv: list[str], exit_code: int
) -> None:
    write_files(hud.cwd / ".hud", files)

    result = hud("project", "use", "browser-evals", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert {path.name: path.read_text() for path in (hud.cwd / ".hud").iterdir()} == files


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
    platform: Platform,
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


def test_a_default_project_that_does_not_exist_is_not_found(
    hud: Hud, platform: Platform, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT="nope")

    result = hud("project", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result


def test_project_list_reads_every_page(hud: Hud, platform: Platform) -> None:
    platform.page_size = 2
    platform.projects += [
        {"id": f"dddddddd-0000-4000-8000-{n:012d}", "name": f"p{n}"} for n in range(3)
    ]

    quiet = hud("project", "list", "--quiet")

    assert quiet.exit_code == 0, quiet
    assert quiet.stdout.split() == [project["id"] for project in platform.projects]


def test_project_list_marks_the_default_and_read_only_projects(
    hud: Hud, platform: Platform
) -> None:
    result = hud("project", "list", "--json")

    assert result.exit_code == 0, result
    assert [(p["name"], p["is_default"], p["can_create"]) for p in result.json] == [
        ("default", True, True),
        ("browser-evals", False, True),
        ("locked-down", False, False),
    ]


@pytest.mark.parametrize(
    ("flags", "linked"), [(["--description", "UI tasks"], True), (["--no-use"], False)]
)
def test_project_create_makes_the_project_and_links_unless_told_not_to(
    hud: Hud, platform: Platform, flags: list[str], linked: bool
) -> None:
    result = hud("project", "create", "UI Evals", *flags, "--yes", "--json")

    assert result.exit_code == 0, result
    (created,) = [project for project in platform.projects if project["name"] == "UI Evals"]
    assert result.json["id"] == created["id"]
    assert created.get("description") == ("UI tasks" if linked else None)
    assert (link(hud) or {}).get("project_id") == (created["id"] if linked else None)


def test_a_refused_project_creation_links_nothing(hud: Hud, platform: Platform) -> None:
    platform.services.route(
        "api", "POST", "/v2/projects", Reply(status=403, json={"detail": "Not enabled"})
    )

    result = hud("project", "create", "UI Evals", "--yes", "--json")

    assert result.exit_code == 1, result
    assert link(hud) is None


def test_using_a_read_only_project_links_nothing(hud: Hud, platform: Platform) -> None:
    result = hud("project", "use", "locked-down", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "permission_denied"), result
    assert link(hud) is None


def test_the_directory_option_after_the_verb_wins(hud: Hud, platform: Platform) -> None:
    result = hud("project", "-C", "group", "use", BROWSER_PROJECT_ID, "-C", "override")

    assert result.exit_code == 0, result
    linked = [path.parent.parent.name for path in hud.cwd.glob("*/.hud/config.json")]
    assert linked == ["override"]
