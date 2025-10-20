from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.source_hash import compute_source_hash, list_source_files

if TYPE_CHECKING:
    from pathlib import Path


def test_source_hash_changes_with_content(tmp_path: Path):
    env = tmp_path / "env"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "pyproject.toml").write_text("[tool.hud]\n")
    (env / "server").mkdir()
    (env / "server" / "main.py").write_text("print('hi')\n")

    h1 = compute_source_hash(env)
    # Change file content
    (env / "server" / "main.py").write_text("print('bye')\n")
    h2 = compute_source_hash(env)
    assert h1 != h2


def test_list_source_files_sorted(tmp_path: Path):
    env = tmp_path / "env"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "environment").mkdir()
    (env / "environment" / "a.py").write_text("a")
    (env / "environment" / "b.py").write_text("b")

    files = list_source_files(env)
    rels = [str(p.resolve().relative_to(env)).replace("\\", "/") for p in files]
    assert rels == ["Dockerfile", "environment/a.py", "environment/b.py"]
