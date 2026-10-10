"""Fixtures wiring the black-box harness (``tests/harness``) into every test.

Every test runs against an isolated SDK configuration: a temporary home, no
developer credentials, telemetry uploads off, and every HUD service URL pointed
at a closed local port. A test that talks to HUD services asks for the
``services`` fixture, which points the SDK at the fake backend instead.

Tests marked ``e2e`` need something outside the process (a Docker daemon, a
Linux sandbox, a real HUD account) and only run when selected with ``-m``;
they skip themselves when what they need is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import dotenv_values

from tests.harness import FakeDocker, FakeServices, Hud, HudEnv, Models

if TYPE_CHECKING:
    from collections.abc import Iterator

# The developer's or CI's own environment, captured before any test isolates it.
# Only ``live`` tests read credentials from it.
REAL_ENVIRONMENT = dict(os.environ)

SERVICE_URL_VARIABLES = (
    "HUD_API_URL",
    "HUD_TELEMETRY_URL",
    "HUD_GATEWAY_URL",
    "HUD_RUNTIME_URL",
    "HUD_RL_URL",
    "HUD_WEB_URL",
)


def real_hud_api_key() -> str | None:
    """``HUD_API_KEY`` from the real environment or the real ``~/.hud/.env``."""
    if key := REAL_ENVIRONMENT.get("HUD_API_KEY"):
        return key
    home = REAL_ENVIRONMENT.get("HOME")
    if home is None:
        return None
    return dotenv_values(Path(home) / ".hud" / ".env").get("HUD_API_KEY")


@pytest.fixture(autouse=True)
def hud_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HudEnv:
    """The SDK's configuration for this test; change it with ``hud_env.set(...)``."""
    return HudEnv(monkeypatch, tmp_path)


@pytest.fixture(scope="session")
def services_server() -> Iterator[FakeServices]:
    server = FakeServices()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def services(services_server: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The fake HUD backend, with the SDK configured to reach it and no routes declared."""
    services_server.reset()
    hud_env.set(**services_server.env())
    return services_server


@pytest.fixture
def models(services: FakeServices) -> Models:
    """Scripted model providers behind the fake gateway."""
    return Models(services)


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The real ``hud`` CLI, run in a subprocess from ``tmp_path / "work"``."""
    return Hud(tmp_path / "work")


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return FakeDocker(tmp_path / "fake-docker", monkeypatch)


@pytest.fixture
def live(hud_env: HudEnv) -> HudEnv:
    """Configure the SDK for a real HUD account: the real key and the real services."""
    key = real_hud_api_key()
    if key is None:
        pytest.skip("needs HUD_API_KEY")
    hud_env.set(
        HUD_API_KEY=key,
        **{name: REAL_ENVIRONMENT.get(name) for name in SERVICE_URL_VARIABLES},
    )
    return hud_env


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=20)
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


def _sandbox_available() -> bool:
    return sys.platform == "linux" and os.geteuid() == 0 and shutil.which("bwrap") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip selected ``e2e`` tests whose daemon, sandbox, or credentials are missing."""
    del config
    checks = {
        "docker": (_docker_available, "needs a running Docker daemon"),
        "sandbox": (_sandbox_available, "needs Linux, root, and bubblewrap"),
        "live": (lambda: real_hud_api_key() is not None, "needs HUD_API_KEY"),
    }
    available: dict[str, bool] = {}
    for item in items:
        for mark, (check, reason) in checks.items():
            if mark not in item.keywords:
                continue
            if mark not in available:
                available[mark] = check()
            if not available[mark]:
                item.add_marker(pytest.mark.skip(reason=reason))
