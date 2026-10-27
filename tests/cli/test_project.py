"""``hud project use``: choosing the Project a directory places new work in, and the
directory link (``.hud/config.json``) that remembers the choice."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from .conftest import TEAM_ID, USER_ID

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

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
