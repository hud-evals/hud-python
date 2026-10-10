"""Where the SDK's configuration comes from, and what a few settings switch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.settings import Settings
from tests.harness import served

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, HudEnv

SERVICE_URLS = {
    "HUD_TELEMETRY_URL": "https://telemetry.hud.ai/v3/api",
    "HUD_API_URL": "https://api.hud.ai",
    "HUD_WEB_URL": "https://hud.ai",
    "HUD_GATEWAY_URL": "https://inference.hud.ai",
    "HUD_RUNTIME_URL": "https://mcp.hud.ai",
    "HUD_RL_URL": "https://rl.hud.ai",
}


@pytest.mark.parametrize(
    ("process", "project", "user", "sent"),
    [
        pytest.param(None, None, "from-hud-set", "from-hud-set", id="hud-set-only"),
        pytest.param(None, "from-project", "from-hud-set", "from-project", id="project-wins"),
        pytest.param("from-env", "from-project", "from-hud-set", "from-env", id="env-wins"),
    ],
)
def test_the_api_key_comes_from_env_then_project_env_then_hud_set(
    services: FakeServices,
    hud_env: HudEnv,
    hud: Hud,
    process: str | None,
    project: str | None,
    user: str,
    sent: str,
) -> None:
    services.route("api", "GET", "/v2/jobs", json={"items": []})
    assert hud("set", f"HUD_API_KEY={user}").exit_code == 0
    if project is not None:
        (hud.cwd / ".env").write_text(f"HUD_API_KEY={project}\n")
    hud_env.set(HUD_API_KEY=process)

    result = hud("jobs", "list", "--json")

    assert result.exit_code == 0, result
    assert [request.bearer for request in services.requests("api", "GET", "/v2/jobs")] == [sent]


def test_service_urls_default_to_the_hud_platform(hud_env: HudEnv) -> None:
    hud_env.set(**dict.fromkeys(SERVICE_URLS))

    configured = Settings(_env_file=None)

    assert {
        variable: getattr(configured, variable.lower()) for variable in SERVICE_URLS
    } == SERVICE_URLS


@pytest.mark.parametrize(
    ("tracking", "published"),
    [
        pytest.param(None, ["shell", "filetracking"], id="default-on"),
        pytest.param("false", ["shell"], id="disabled"),
    ],
)
async def test_file_tracking_follows_hud_file_tracking_enabled(
    hud_env: HudEnv, tmp_path: Path, tracking: str | None, published: list[str]
) -> None:
    hud_env.set(HUD_FILE_TRACKING_ENABLED=tracking)
    env = Environment("tracked")
    env.workspace(tmp_path / "workspace")

    async with served(env) as client:
        assert client.manifest is not None
        names = [binding.name for binding in client.manifest.bindings]

    assert names == published
