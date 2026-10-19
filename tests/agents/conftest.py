"""Fixtures for the agent contracts: a provider reached directly, the browser-use SDK
stand-in, and the ``driver`` for the ``family`` a test is parametrized with."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

from tests.agents.families import (
    FAMILIES,
    BrowserUseDriver,
    CliDriver,
    CliFamily,
    ToolDriver,
    ToolFamily,
)
from tests.support import FakeServices, Models, browser_use, fake_browser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path
    from types import ModuleType

    from tests.agents.families import Driver
    from tests.support import HudEnv


@pytest.fixture(scope="session")
def provider_server() -> Iterator[FakeServices]:
    server = FakeServices()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def provider(provider_server: FakeServices) -> Models:
    """Scripted models served as a provider's own API (not behind the HUD gateway)."""
    provider_server.reset()
    return Models(provider_server)


@pytest.fixture
def browser_use_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The browser-use SDK stand-in (``tests/support/browser_use.py``), imported by the
    agent as ``browser_use`` in place of any real install."""
    monkeypatch.setitem(sys.modules, "browser_use", browser_use)
    yield browser_use
    browser_use.stopped.clear()


@pytest.fixture
async def driver(
    family: str,
    models: Models,
    provider: Models,
    hud_env: HudEnv,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> AsyncIterator[Driver]:
    """The driver for ``family``, signed in the way the family's agent routes by default:
    a HUD key for the gateway agents, an Anthropic key for browser-use."""
    chosen: Any = FAMILIES[family]
    if isinstance(chosen, ToolFamily):
        hud_env.set(HUD_API_KEY="k")
        yield ToolDriver(chosen, models, provider, tmp_path / "ws")
    elif isinstance(chosen, CliFamily):
        hud_env.set(HUD_API_KEY="k")
        yield CliDriver(chosen, models, tmp_path)
    else:
        request.getfixturevalue("browser_use_sdk")
        hud_env.set(ANTHROPIC_API_KEY="anthropic-key", ANTHROPIC_BASE_URL=provider.url)
        async with fake_browser() as browser:
            yield BrowserUseDriver(chosen, models, provider, tmp_path / "ws", browser)
