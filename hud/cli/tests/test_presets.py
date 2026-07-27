"""Tests for materializing ``hud init`` starter presets."""

from __future__ import annotations

import io
import tarfile
import time
from typing import TYPE_CHECKING

import pytest

from hud.cli import presets as presets_module
from hud.cli.presets import PRESETS_BY_ID, EnvironmentPreset, materialize_preset

if TYPE_CHECKING:
    from pathlib import Path


def _tarball(entries: dict[str, tuple[bytes, int]]) -> bytes:
    """Build a GitHub-shaped ``<repo>-main/`` archive from name → (data, mode)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, (data, mode) in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_vendored_preset_is_copied_from_the_installed_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_network(preset: EnvironmentPreset) -> bytes:
        raise AssertionError("a vendored preset must not download anything")

    monkeypatch.setattr(presets_module, "_download_tarball", _no_network)

    target = tmp_path / "coding"
    materialize_preset(PRESETS_BY_ID["coding"], target)

    assert (target / "env.py").exists()
    assert (target / "Dockerfile.hud").exists()
    assert (target / "coding" / "repo.py").exists()
    assert (target / "tests" / "conftest.py").exists()


def test_vendored_preset_copy_preserves_executable_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starters = tmp_path / "starters"
    (starters / "blank" / "scripts").mkdir(parents=True)
    script = starters / "blank" / "scripts" / "entrypoint.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    (starters / "blank" / "env.py").write_text("env = None\n")
    monkeypatch.setattr(presets_module, "_STARTERS_DIR", starters)

    target = tmp_path / "out"
    materialize_preset(PRESETS_BY_ID["blank"], target)

    assert (target / "env.py").read_text() == "env = None\n"
    assert (target / "scripts" / "entrypoint.sh").stat().st_mode & 0o111


def test_vendored_preset_copy_skips_checkout_build_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starters = tmp_path / "starters"
    (starters / "blank").mkdir(parents=True)
    (starters / "blank" / "env.py").write_text("env = None\n")
    (starters / "blank" / ".venv" / "bin").mkdir(parents=True)
    (starters / "blank" / ".venv" / "bin" / "python").write_text("")
    (starters / "blank" / "__pycache__").mkdir()
    (starters / "blank" / "__pycache__" / "env.cpython-311.pyc").write_text("")
    monkeypatch.setattr(presets_module, "_STARTERS_DIR", starters)

    target = tmp_path / "out"
    materialize_preset(PRESETS_BY_ID["blank"], target)

    assert (target / "env.py").exists()
    assert not (target / ".venv").exists()
    assert not (target / "__pycache__").exists()


def test_missing_vendored_tree_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(presets_module, "_STARTERS_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(presets_module, "_CHECKOUT_STARTERS_DIR", tmp_path / "also-nowhere")

    with pytest.raises(FileNotFoundError):
        materialize_preset(PRESETS_BY_ID["cua"], tmp_path / "out")


def test_remote_preset_extracts_the_repo_tarball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _tarball(
        {
            "hud-browser-main/README.md": (b"# browser", 0o644),
            "hud-browser-main/scripts/run.sh": (b"#!/bin/sh\n", 0o755),
        }
    )
    monkeypatch.setattr(presets_module, "_download_tarball", lambda preset: payload)

    target = tmp_path / "browser"
    materialize_preset(PRESETS_BY_ID["browser"], target)

    assert (target / "README.md").read_text() == "# browser"
    assert (target / "scripts" / "run.sh").stat().st_mode & 0o111


def test_remote_preset_refuses_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _tarball({"hud-browser-main/../escaped.txt": (b"nope", 0o644)})
    monkeypatch.setattr(presets_module, "_download_tarball", lambda preset: payload)

    with pytest.raises(ValueError, match="unsafe path"):
        materialize_preset(PRESETS_BY_ID["browser"], tmp_path / "browser")

    assert not (tmp_path / "escaped.txt").exists()
