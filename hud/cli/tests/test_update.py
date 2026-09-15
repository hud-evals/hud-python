"""Tests for the PyPI update banner."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from hud.cli import update


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HUD_SKIP_VERSION_CHECK", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(update, "__version__", "1.0.0")


def _pypi(version: str) -> httpx.Response:
    return httpx.Response(200, json={"info": {"version": version}})


def _fail_fetch(*_a: object, **_k: object) -> httpx.Response:
    raise AssertionError("fetch")


def test_outdated_prints_banner_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fetches = {"n": 0}

    def get(*_a: object, **_k: object) -> httpx.Response:
        fetches["n"] += 1
        return _pypi("2.0.0")

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(update.sys, "prefix", "/usr")
    monkeypatch.setattr(update.sys, "base_prefix", "/usr")
    update.notify_if_outdated(["hud", "eval"])
    first = capsys.readouterr().err
    assert "2.0.0" in first
    assert "current: 1.0.0" in first
    assert "uv tool upgrade hud" in first
    update.notify_if_outdated(["hud", "eval"])
    assert capsys.readouterr().err.count("2.0.0") == 1
    assert fetches["n"] == 1
    cached = json.loads((tmp_path / ".hud" / ".cache" / "version_check.json").read_text())
    assert cached["latest"] == "2.0.0"


def test_current_version_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("1.0.0"))
    update.notify_if_outdated(["hud", "eval"])
    assert capsys.readouterr().err == ""


def test_project_venv_suggests_uv_sync(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
    monkeypatch.setattr(update.sys, "prefix", "/proj/.venv")
    monkeypatch.setattr(update.sys, "base_prefix", "/usr")
    update.notify_if_outdated(["hud", "eval"])
    assert "uv sync --upgrade-package hud" in capsys.readouterr().err


def test_uv_tool_install_is_not_a_project_venv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
    monkeypatch.setattr(update.sys, "prefix", "/Users/x/.local/share/uv/tools/hud")
    monkeypatch.setattr(update.sys, "base_prefix", "/usr")
    update.notify_if_outdated(["hud", "eval"])
    assert "uv tool upgrade hud" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [["hud"], ["hud", "--version"], ["hud", "--help"], ["hud", "--version", "--json"]],
)
def test_help_and_version_do_not_fetch(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", _fail_fetch)
    update.notify_if_outdated(argv)


def test_ci_and_opt_out_do_not_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", _fail_fetch)
    monkeypatch.setenv("CI", "1")
    update.notify_if_outdated(["hud", "eval"])
    monkeypatch.delenv("CI")
    monkeypatch.setenv("HUD_SKIP_VERSION_CHECK", "1")
    update.notify_if_outdated(["hud", "eval"])


def test_expired_cache_refetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / ".hud" / ".cache" / "version_check.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"latest": "1.5.0", "checked_at": 0.0}))
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _pypi("2.0.0"))
    update.notify_if_outdated(["hud", "eval"])
    assert "2.0.0" in capsys.readouterr().err


def test_fetch_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def down(*_a: object, **_k: object) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", down)
    update.notify_if_outdated(["hud", "eval"])
    assert capsys.readouterr().err == ""
