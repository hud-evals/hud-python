from __future__ import annotations

import sys

import pytest

from hud.cli.utils import docker as mod

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Prefers Linux")


def test_emit_docker_hints_windows(monkeypatch):
    # Patch the global hud_console used by hint printing

    fake = type(
        "C",
        (),
        {
            "error": lambda *a, **k: None,
            "hint": lambda *a, **k: None,
            "dim_info": lambda *a, **k: None,
        },
    )()
    monkeypatch.setattr("hud.utils.hud_console.hud_console", fake, raising=False)
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    mod._emit_docker_hints("cannot connect to the docker daemon")


def test_emit_docker_hints_linux(monkeypatch):
    fake = type(
        "C",
        (),
        {
            "error": lambda *a, **k: None,
            "hint": lambda *a, **k: None,
            "dim_info": lambda *a, **k: None,
        },
    )()
    monkeypatch.setattr("hud.utils.hud_console.hud_console", fake, raising=False)
    monkeypatch.setattr(mod.platform, "system", lambda: "Linux")
    mod._emit_docker_hints("Cannot connect to the Docker daemon")


def test_emit_docker_hints_darwin(monkeypatch):
    fake = type(
        "C",
        (),
        {
            "error": lambda *a, **k: None,
            "hint": lambda *a, **k: None,
            "dim_info": lambda *a, **k: None,
        },
    )()
    monkeypatch.setattr("hud.utils.hud_console.hud_console", fake, raising=False)
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")
    mod._emit_docker_hints("error during connect: is the docker daemon running")


def test_emit_docker_hints_generic(monkeypatch):
    fake = type(
        "C",
        (),
        {
            "error": lambda *a, **k: None,
            "hint": lambda *a, **k: None,
            "dim_info": lambda *a, **k: None,
        },
    )()
    monkeypatch.setattr("hud.utils.hud_console.hud_console", fake, raising=False)
    monkeypatch.setattr(mod.platform, "system", lambda: "Other")
    mod._emit_docker_hints("some unrelated error")
