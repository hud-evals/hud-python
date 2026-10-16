"""The names each package exports, and the CLI's identity."""

from __future__ import annotations

import importlib

import pytest

PACKAGES = [
    "hud",
    "hud.environment",
    "hud.environment.robot",
    "hud.eval",
    "hud.eval.runtime",
    "hud.agents",
    "hud.agents.robot",
    "hud.agents.tools",
    "hud.capabilities",
    "hud.clients",
    "hud.graders",
    "hud.integrations.harbor",
    "hud.telemetry",
    "hud.train",
    "hud.utils",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exported_name_resolves(package: str) -> None:
    module = importlib.import_module(package)

    assert [name for name in module.__all__ if not hasattr(module, name)] == []
