"""Tests for packing Harbor folders into an import bundle."""

from __future__ import annotations

import os
import zipfile
from typing import TYPE_CHECKING

import pytest

from hud.integrations.harbor.importing import BundleError, pack_bundle

if TYPE_CHECKING:
    from pathlib import Path


def _task(root: Path, name: str = "hello") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text("version = '1.0'\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu\n")
    return task


def _members(bundle: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(bundle) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_each_folder_is_stored_under_its_own_name(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _task(dataset, "one")
    _task(dataset, "two")
    bundle = tmp_path / "bundle.zip"

    assert pack_bundle([dataset, _task(tmp_path / "loose")], bundle) == 6

    assert sorted(_members(bundle)) == [
        "dataset/one/environment/Dockerfile",
        "dataset/one/task.toml",
        "dataset/two/environment/Dockerfile",
        "dataset/two/task.toml",
        "hello/environment/Dockerfile",
        "hello/task.toml",
    ]


def test_symlinked_files_are_stored_by_content_and_symlinked_folders_are_not_followed(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path / "src")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "data.txt").write_text("shared")
    os.symlink(shared / "data.txt", task / "environment" / "data.txt")
    os.symlink(shared, task / "environment" / "linked")
    os.symlink(tmp_path / "missing", task / "dangling")
    bundle = tmp_path / "bundle.zip"

    pack_bundle([task], bundle)

    members = _members(bundle)
    assert members["hello/environment/data.txt"] == b"shared"
    assert not any("linked" in name or "dangling" in name for name in members)


def test_two_folders_with_one_name_are_refused(tmp_path: Path) -> None:
    first = _task(tmp_path / "a")
    second = _task(tmp_path / "b")

    with pytest.raises(BundleError, match="both named 'hello'"):
        pack_bundle([first, second], tmp_path / "bundle.zip")


def test_folders_without_harbor_files_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "README.md").write_text("hi")

    with pytest.raises(BundleError, match="no Harbor task folder"):
        pack_bundle([folder], tmp_path / "bundle.zip")


def test_a_path_that_is_not_a_folder_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "task.toml"
    file.write_text("")

    with pytest.raises(BundleError, match="is not a folder"):
        pack_bundle([file], tmp_path / "bundle.zip")
