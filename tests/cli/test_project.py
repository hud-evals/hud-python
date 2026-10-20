"""``hud project``: which Project a directory places new work in, choosing one, creating
one, and the directory link (``.hud/config.json``) that remembers the choice."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.services import Reply

from .conftest import TEAM_ID, USER_ID

if TYPE_CHECKING:
    from tests.conftest import Hud, HudEnv
    from tests.fixtures.services import FakeServices, Request

DEFAULT = "aaaaaaaa-0000-4000-8000-000000000001"
BROWSER = "aaaaaaaa-0000-4000-8000-000000000002"
LOCKED = "aaaaaaaa-0000-4000-8000-000000000003"
OTHER_ID = "cccccccc-0000-4000-8000-000000000001"
PROJECTS = [
    {"id": DEFAULT, "name": "default", "is_default": True, "capabilities": {"create": True}},
    {"id": BROWSER, "name": "browser-evals", "capabilities": {"create": True}},
    {"id": LOCKED, "name": "locked-down", "capabilities": {}},
]


@pytest.fixture
def projects(signed_in: FakeServices) -> FakeServices:
    """The platform listing ``PROJECTS`` and answering each by id."""
    signed_in.route("api", "GET", "/v2/projects", json={"items": PROJECTS, "total": 3})
    for project in PROJECTS:
        signed_in.route("api", "GET", f"/v2/projects/{project['id']}", json=project)
    return signed_in


def linked(hud: Hud) -> dict[str, Any] | None:
    path = hud.cwd / ".hud" / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.parametrize(
    ("ref", "project"),
    [("browser-evals", BROWSER), ("Browser Evals", BROWSER), (BROWSER.upper(), BROWSER)],
    ids=["by-name", "by-display-name", "by-an-upper-case-id"],
)
def test_use_links_the_directory_to_the_project_under_these_credentials(
    ref: str, project: str, projects: FakeServices, hud: Hud
) -> None:
    result = hud("project", "use", ref, "--json")

    assert (result.exit_code, result.json["id"]) == (0, project), result
    stored = linked(hud)
    assert stored is not None
    assert (stored["project_id"], stored["scope"]["user_id"], stored["scope"]["team_id"]) == (
        project,
        USER_ID,
        TEAM_ID,
    )
    assert "sk-hud-test" not in json.dumps(stored)


def test_use_with_dry_run_links_nothing(projects: FakeServices, hud: Hud) -> None:
    result = hud("project", "use", "browser-evals", "--dry-run", "--json")

    assert (result.exit_code, result.json["dry_run"], linked(hud)) == (0, True, None), result


@pytest.mark.parametrize(
    ("ref", "error"),
    [("locked-down", "permission_denied"), ("nowhere", "not_found")],
    ids=["a-read-only-project", "an-unknown-name"],
)
def test_use_refuses_a_project_it_cannot_place_work_in(
    ref: str, error: str, projects: FakeServices, hud: Hud
) -> None:
    result = hud("project", "use", ref, "--json")

    assert (result.exit_code, result.json["error"], linked(hud)) == (1, error, None), result


def test_a_name_two_projects_share_must_be_given_as_an_id(
    signed_in: FakeServices, hud: Hud
) -> None:
    twins = [{**PROJECTS[1], "id": DEFAULT}, PROJECTS[1]]
    signed_in.route("api", "GET", "/v2/projects", json={"items": twins, "total": 2})

    result = hud("project", "use", "browser-evals", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert DEFAULT in result.json["message"]
    assert BROWSER in result.json["message"]


@pytest.mark.parametrize(
    "scope",
    [
        {"origin": "SAME", "user_id": OTHER_ID, "team_id": TEAM_ID},
        {"origin": "SAME", "user_id": USER_ID, "team_id": OTHER_ID},
        {"origin": "https://api.example", "user_id": USER_ID, "team_id": TEAM_ID},
    ],
    ids=["by-another-user", "in-another-team", "on-another-platform"],
)
def test_a_link_made_under_other_credentials_is_left_alone(
    scope: dict[str, str], projects: FakeServices, hud: Hud
) -> None:
    origin = projects.url("api").removesuffix("/api")
    link = {"version": 1, "scope": {**scope, "origin": scope["origin"].replace("SAME", origin)}}
    (hud.cwd / ".hud").mkdir()
    (hud.cwd / ".hud" / "config.json").write_text(json.dumps({**link, "project_id": DEFAULT}))
    before = (hud.cwd / ".hud" / "config.json").read_text()

    result = hud("project", "use", "browser-evals", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "failure"), result
    assert (hud.cwd / ".hud" / "config.json").read_text() == before


def test_a_corrupt_link_is_refused_and_left_alone(projects: FakeServices, hud: Hud) -> None:
    (hud.cwd / ".hud").mkdir()
    (hud.cwd / ".hud" / "config.json").write_text('{"broken":')

    result = hud("project", "use", "browser-evals", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "failure"), result
    assert (hud.cwd / ".hud" / "config.json").read_text() == '{"broken":'


def test_an_older_camelcase_link_keeps_its_ids_when_rewritten(
    projects: FakeServices, hud: Hud
) -> None:
    (hud.cwd / ".hud").mkdir()
    (hud.cwd / ".hud" / "config.json").write_text(
        json.dumps({"registryId": OTHER_ID, "tasksetId": DEFAULT, "registryName": "old"})
    )

    result = hud("project", "use", "browser-evals", "--json")

    stored = linked(hud)
    assert result.exit_code == 0, result
    assert stored is not None
    assert (stored["registry_id"], stored["taskset_id"], stored["project_id"]) == (
        OTHER_ID,
        DEFAULT,
        BROWSER,
    )
    assert "registryName" not in stored


@pytest.mark.parametrize(
    ("link", "default", "project", "source"),
    [
        (None, None, None, "team default"),
        (None, "Browser Evals", BROWSER, "HUD_DEFAULT_PROJECT"),
        (BROWSER, "locked-down", BROWSER, ".hud/config.json"),
    ],
    ids=["the-team-default", "hud-default-project", "the-link-beats-hud-default-project"],
)
def test_project_shows_the_project_the_directory_uses_and_why(
    link: str | None,
    default: str | None,
    project: str | None,
    source: str,
    projects: FakeServices,
    hud_env: HudEnv,
    hud: Hud,
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT=default)
    if link is not None:
        (hud.cwd / ".hud").mkdir()
        (hud.cwd / ".hud" / "config.json").write_text(json.dumps({"project_id": link}))

    result = hud("project", "--json")

    assert result.exit_code == 0, result
    assert ((result.json["project"] or {}).get("id"), result.json["source"]) == (project, source)


def test_a_default_project_that_does_not_exist_is_not_found(
    projects: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_DEFAULT_PROJECT="nope")

    result = hud("project", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result


def test_the_directory_option_after_the_verb_wins(projects: FakeServices, hud: Hud) -> None:
    result = hud("project", "-C", "group", "use", BROWSER, "-C", "override")

    assert result.exit_code == 0, result
    assert [p.parent.parent.name for p in hud.cwd.glob("*/.hud/config.json")] == ["override"]


def test_list_reads_every_page_and_marks_the_default_and_read_only_projects(
    signed_in: FakeServices, hud: Hud
) -> None:
    def one_a_page(request: Request) -> Reply:
        offset = int(request.query["offset"][0])
        return Reply(json={"items": PROJECTS[offset : offset + 1], "total": len(PROJECTS)})

    signed_in.route("api", "GET", "/v2/projects", handler=one_a_page)

    listed = hud("project", "list", "--json")

    assert [(p["name"], p["is_default"], p["can_create"]) for p in listed.json] == [
        ("default", True, True),
        ("browser-evals", False, True),
        ("locked-down", False, False),
    ]


@pytest.mark.parametrize(
    ("options", "description", "links"),
    [([], None, True), (["--description", "UI tasks", "--no-use"], "UI tasks", False)],
    ids=["links-the-directory", "without-linking"],
)
def test_create_makes_the_project_and_links_the_directory_unless_told_not_to(
    options: list[str], description: str | None, links: bool, signed_in: FakeServices, hud: Hud
) -> None:
    created = {"id": OTHER_ID, "name": "ui-evals", "capabilities": {"create": True}}
    signed_in.route("api", "POST", "/v2/projects", json=created)

    result = hud("project", "create", "UI Evals", *options, "--yes", "--json")

    assert (result.exit_code, result.json["id"]) == (0, OTHER_ID), result
    assert signed_in.bodies("api", "POST", "/v2/projects") == [
        {"name": "UI Evals", **({"description": description} if description else {})}
    ]
    assert ((linked(hud) or {}).get("project_id") == OTHER_ID) is links
