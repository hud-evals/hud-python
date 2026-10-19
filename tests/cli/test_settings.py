"""Where the SDK's configuration comes from, and what a few settings switch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.support import Hud, HudEnv, Platform


@pytest.mark.parametrize(
    ("process", "project", "user", "sent"),
    [
        pytest.param(None, None, "from-hud-set", "from-hud-set", id="hud-set-only"),
        pytest.param(None, "from-project", "from-hud-set", "from-project", id="project-wins"),
        pytest.param("from-env", "from-project", "from-hud-set", "from-env", id="env-wins"),
    ],
)
def test_the_api_key_comes_from_env_then_project_env_then_hud_set(
    platform: Platform,
    hud_env: HudEnv,
    hud: Hud,
    process: str | None,
    project: str | None,
    user: str,
    sent: str,
) -> None:
    platform.keys = {sent}
    assert hud("set", f"HUD_API_KEY={user}").exit_code == 0
    if project is not None:
        (hud.cwd / ".env").write_text(f"HUD_API_KEY={project}\n")
    hud_env.set(HUD_API_KEY=process)

    result = hud("jobs", "list", "--json")

    assert result.exit_code == 0, result
