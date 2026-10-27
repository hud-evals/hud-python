"""Where the CLI's configuration comes from: the process environment, then the project's
``.env``, then what ``hud set`` saved."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import Hud, HudEnv
    from tests.fixtures.services import FakeServices


@pytest.mark.parametrize(
    ("process", "project", "sent"),
    [
        (None, None, "from-hud-set"),
        (None, "from-project", "from-project"),
        ("from-env", "from-project", "from-env"),
    ],
    ids=["hud-set-only", "the-project-env-wins", "the-process-env-wins"],
)
def test_the_api_key_comes_from_the_environment_then_the_project_then_hud_set(
    process: str | None,
    project: str | None,
    sent: str,
    services: FakeServices,
    hud_env: HudEnv,
    hud: Hud,
) -> None:
    services.route("api", "GET", "/v2/jobs", json={"items": [], "total": 0})
    assert hud("set", "HUD_API_KEY=from-hud-set").exit_code == 0
    if project is not None:
        (hud.cwd / ".env").write_text(f"HUD_API_KEY={project}\n")
    hud_env.set(HUD_API_KEY=process)

    result = hud("jobs", "list", "--json")

    assert result.exit_code == 0, result
    assert [request.bearer for request in services.requests("api", "GET", "/v2/jobs")] == [sent]
