"""The signed-in platform ``hud`` scenarios start from, and its Projects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.harness import Hud, Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, HudEnv, Request

API_KEY = "sk-hud-test"
USER_ID = "11111111-1111-4111-8111-111111111111"
TEAM_ID = "22222222-2222-4222-8222-222222222222"
WEB_URL = "https://hud.example"


@pytest.fixture
def platform(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The fake platform with a signed-in user: a HUD key and ``GET /auth/me``.

    The web app is only printed as links, so it keeps a fixed address.
    """
    hud_env.set(HUD_API_KEY=API_KEY, HUD_WEB_URL=WEB_URL)
    services.route("api", "GET", "/v2/auth/me", json={"user_id": USER_ID, "team_id": TEAM_ID})
    return services


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The ``hud`` fixture at a terminal width whose snapshots fit the line limit."""
    return Hud(tmp_path / "work", columns=96)


DEFAULT_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000001"
BROWSER_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000002"
LOCKED_PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000003"
PROJECTS = [
    {
        "id": DEFAULT_PROJECT_ID,
        "name": "default",
        "is_default": True,
        "capabilities": {"view": True, "create": True},
    },
    {"id": BROWSER_PROJECT_ID, "name": "browser-evals", "capabilities": {"create": True}},
    {"id": LOCKED_PROJECT_ID, "name": "locked-down", "capabilities": {"view": True}},
]


def list_projects(request: Request) -> Reply:
    """``GET /projects``: a name search matches substrings, as the platform's does."""
    search = request.query.get("search", [""])[0]
    offset = int(request.query.get("offset", ["0"])[0])
    matches = [project for project in PROJECTS if search in project["name"]]
    return Reply(json={"items": matches[offset:], "total": len(matches)})


def get_project(request: Request) -> Reply:
    found = [project for project in PROJECTS if project["id"] == request.params["id"]]
    return Reply(json=found[0]) if found else Reply(status=404, json={"detail": "Not found"})


@pytest.fixture
def projects(platform: FakeServices) -> FakeServices:
    """The signed-in team's Projects: its default, a writable one and a read-only one."""
    platform.route("api", "GET", "/v2/projects", handler=list_projects)
    platform.route("api", "GET", "/v2/projects/{id}", handler=get_project)
    return platform
