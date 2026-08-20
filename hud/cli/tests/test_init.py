"""Tests for ``hud init`` example environment materialization."""

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest
import typer

from hud.cli import init as init_module
from hud.cli import presets as presets_module
from hud.cli.init import init_command
from hud.cli.presets import PRESETS_BY_ID, EnvironmentPreset, materialize_preset

if TYPE_CHECKING:
    from pathlib import Path


def _fake_materialize(record: dict[str, object]):
    def materialize(preset: EnvironmentPreset, target: Path) -> None:
        record["preset"] = preset
        record["target"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text("# example environment")
        (target / "env.py").write_text(f'env = Environment(name="{preset.id}")\n')

    return materialize


def _sdk_archive(source: str, files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"hud-python-release/environments/{source}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return payload.getvalue()


def test_init_uses_coding_example_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {}
    monkeypatch.setattr(init_module, "materialize_preset", _fake_materialize(record))

    init_command(name="my-cool-env", directory=str(tmp_path), force=False, preset=None)

    target = tmp_path / "my-cool-env"
    assert (target / "README.md").read_text() == "# example environment"
    assert record["target"] == target
    assert record["preset"] == PRESETS_BY_ID["coding"]
    assert 'Environment(name="my-cool-env")' in (target / "env.py").read_text()


def test_init_blank_materializes_runnable_example(tmp_path: Path) -> None:
    init_command(name="berry", directory=str(tmp_path), force=False, preset="blank")

    target = tmp_path / "berry"
    assert {path.name for path in target.iterdir()} == {
        "README.md",
        "pyproject.toml",
        "env.py",
        "tasks.py",
        "Dockerfile.hud",
        ".dockerignore",
    }
    assert 'Environment(name="berry")' in (target / "env.py").read_text()
    assert "package = false" in (target / "pyproject.toml").read_text()
    assert 'CMD ["/usr/local/venv/bin/hud", "serve"' in (target / "Dockerfile.hud").read_text()
    assert ".venv" in (target / ".dockerignore").read_text()


def test_init_blank_uses_normalized_environment_name(tmp_path: Path) -> None:
    init_command(name="My Cool_Env", directory=str(tmp_path), force=False, preset="blank")

    target = tmp_path / "My Cool_Env"
    assert 'Environment(name="my-cool-env")' in (target / "env.py").read_text()
    assert 'name = "blank-env"' in (target / "pyproject.toml").read_text()


def test_init_refuses_to_clobber_nonempty_directory(tmp_path: Path) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "precious.txt").write_text("data")

    with pytest.raises(typer.Exit):
        init_command(name="taken", directory=str(tmp_path), force=False, preset="blank")

    assert (target / "precious.txt").read_text() == "data"


def test_init_force_overwrites_existing_blank_files(tmp_path: Path) -> None:
    target = tmp_path / "env"
    target.mkdir()
    (target / "env.py").write_text("old")

    init_command(name="env", directory=str(tmp_path), force=True, preset="blank")

    assert "Environment" in (target / "env.py").read_text()


def test_init_without_name_uses_example_id_as_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {}
    monkeypatch.setattr(init_module, "materialize_preset", _fake_materialize(record))

    init_command(name=None, directory=str(tmp_path), force=False, preset="coding")

    target = tmp_path / "coding"
    assert (target / "README.md").read_text() == "# example environment"
    assert record["target"] == target


def test_init_name_overrides_example_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {}
    monkeypatch.setattr(init_module, "materialize_preset", _fake_materialize(record))

    init_command(name="custom", directory=str(tmp_path), force=False, preset="cua")

    assert (tmp_path / "custom" / "README.md").exists()
    assert 'Environment(name="custom")' in (tmp_path / "custom" / "env.py").read_text()
    assert not (tmp_path / "cua").exists()


def test_init_without_name_or_example_errors_when_noninteractive(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        init_command(name=None, directory=str(tmp_path), force=False, preset=None)


def test_init_rejects_unknown_example(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        init_command(name=None, directory=str(tmp_path), force=False, preset="does-not-exist")


def test_materialize_preset_copies_local_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    source = repository / "environments" / "coding"
    source.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'hud'\n")
    (source / "env.py").write_text("env = object()")
    (source / ".venv").mkdir()
    (source / ".venv" / "ignored").write_text("ignored")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")

    target = tmp_path / "project"
    monkeypatch.setattr(
        presets_module,
        "__file__",
        str(repository / "hud" / "cli" / "presets.py"),
    )
    materialize_preset(PRESETS_BY_ID["coding"], target)

    assert (target / "env.py").read_text() == "env = object()"
    assert not (target / ".venv").exists()
    assert not (target / "__pycache__").exists()


def test_materialize_preset_extracts_matching_sdk_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _sdk_archive(
        "coding",
        {
            "README.md": b"# Coding",
            "scripts/run.sh": b"#!/bin/sh\n",
        },
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = httpx.Response(
        200,
        content=payload,
        request=httpx.Request("GET", "https://example.test"),
    )
    monkeypatch.setattr(presets_module.httpx, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(
        presets_module,
        "__file__",
        str(tmp_path / "installed" / "hud" / "cli" / "presets.py"),
    )
    monkeypatch.setattr(presets_module, "__version__", "1.2.3")

    target = tmp_path / "project"
    materialize_preset(PRESETS_BY_ID["coding"], target)

    requested_url = client.get.call_args.args[0]
    assert requested_url.endswith("/tar.gz/refs/tags/v1.2.3")
    assert (target / "README.md").read_text() == "# Coding"
    assert (target / "scripts" / "run.sh").read_text() == "#!/bin/sh\n"


def test_materialize_preset_rejects_unsafe_archive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _sdk_archive("coding", {"../../escape": b"unsafe"})
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = httpx.Response(
        200,
        content=payload,
        request=httpx.Request("GET", "https://example.test"),
    )
    monkeypatch.setattr(presets_module.httpx, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(
        presets_module,
        "__file__",
        str(tmp_path / "installed" / "hud" / "cli" / "presets.py"),
    )
    monkeypatch.setattr(presets_module, "__version__", "1.2.3")

    with pytest.raises(ValueError, match="unsafe path"):
        materialize_preset(PRESETS_BY_ID["coding"], tmp_path / "project")

    assert not (tmp_path / "escape").exists()


def test_materialize_preset_requires_checkout_for_development_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        presets_module,
        "__file__",
        str(tmp_path / "installed" / "hud" / "cli" / "presets.py"),
    )
    monkeypatch.setattr(presets_module, "__version__", "1.2.3.dev0")
    with pytest.raises(ValueError, match="development version"):
        materialize_preset(PRESETS_BY_ID["coding"], tmp_path / "project")
