"""``hud.cli.utils.version_check`` — version compare, cache round-trip, PyPI fetch,
and the update prompt, with network + cache fully mocked.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from hud.cli.utils import version_check as vc
from hud.cli.utils.version_check import VersionInfo
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_compare_versions() -> None:
    assert vc._compare_versions("1.0.0", "1.0.1") is True
    assert vc._compare_versions("1.0.1", "1.0.0") is False
    assert vc._compare_versions("unknown", "2.0.0") is False


def test_current_version_and_virtualenv_are_typed() -> None:
    assert isinstance(vc._get_current_version(), str)
    assert isinstance(vc._is_in_virtualenv(), bool)


def _point_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vc, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(vc, "VERSION_CACHE_FILE", tmp_path / ".cache" / "version_check.json")


def test_cache_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_cache(monkeypatch, tmp_path)
    info = VersionInfo(latest="1.1.0", current="1.0.0", is_outdated=True, checked_at=time.time())

    vc._save_cache(info)
    loaded = vc._load_cache()

    assert loaded is not None
    assert loaded.latest == "1.1.0"
    assert loaded.current == "1.0.0"


def test_expired_cache_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_cache(monkeypatch, tmp_path)
    stale = VersionInfo(latest="1.1.0", current="1.0.0", is_outdated=True, checked_at=0.0)
    vc._save_cache(stale)
    assert vc._load_cache() is None


def test_missing_cache_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_cache(monkeypatch, tmp_path)
    assert vc._load_cache() is None


def test_fetch_latest_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"info": {"version": "9.9.9"}}

    class FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...
        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_a: Any) -> bool:
            return False

        def get(self, _url: str) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr(vc.httpx, "Client", FakeClient)
    assert vc._fetch_latest_version() == "9.9.9"


def test_check_for_updates_fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_cache(monkeypatch, tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HUD_SKIP_VERSION_CHECK", raising=False)
    monkeypatch.setattr(vc, "_get_current_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "_fetch_latest_version", lambda: "2.0.0")

    info = vc.check_for_updates()

    assert info is not None
    assert info.latest == "2.0.0"
    assert info.is_outdated is True
    # The fresh check should have written a cache file.
    assert vc.VERSION_CACHE_FILE.exists()


def test_check_for_updates_skipped_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "1")
    assert vc.check_for_updates() is None


def test_display_update_prompt_outdated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vc,
        "check_for_updates",
        lambda: VersionInfo(latest="2.0.0", current="1.0.0", is_outdated=True, checked_at=0.0),
    )
    # Should render without raising.
    vc.display_update_prompt(HUDConsole())


def test_force_version_check_clears_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_cache(monkeypatch, tmp_path)
    vc._save_cache(VersionInfo("1.1.0", "1.0.0", True, time.time()))
    assert vc.VERSION_CACHE_FILE.exists()
    monkeypatch.setattr(vc, "check_for_updates", lambda: None)

    vc.force_version_check()

    assert not vc.VERSION_CACHE_FILE.exists()
