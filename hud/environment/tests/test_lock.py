"""The ``hud.lock.yaml`` format: round-trip, fingerprint, build composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.environment import lock
from hud.environment.source import EnvironmentSource

if TYPE_CHECKING:
    from pathlib import Path


def test_write_read_and_fingerprint(tmp_path: Path) -> None:
    lock_path = tmp_path / "hud.lock.yaml"
    lock_data = {"version": "2.0", "build": {"version": "0.1.0"}}

    written = lock.write_lock(lock_path, lock_data)
    digest, size = lock.lock_fingerprint(lock_data)

    assert written == lock_path
    assert lock.read_lock(written) == lock_data
    assert len(digest) == 64
    assert size == len(lock.dump_lock(lock_data, sort_keys=True))


def test_local_image_prefers_images_local_over_legacy_image() -> None:
    assert lock.local_image({"images": {"local": "env:1.0"}, "image": "old"}) == "env:1.0"
    assert lock.local_image({"image": "old:1"}) == "old:1"
    assert lock.local_image({}) == ""


def test_build_lock_data_builds_shared_lock_shape(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile.hud").write_text(
        "FROM python:3.11\nENV OPENAI_API_KEY=\n",
        encoding="utf-8",
    )
    controller_dir = tmp_path / "controller"
    controller_dir.mkdir()
    (controller_dir / "server.py").write_text("print('ok')\n", encoding="utf-8")

    capability = {"name": "shell", "protocol": "ssh/2", "url": "ssh://host:22", "params": {}}
    lock_data = lock.build_lock_data(
        EnvironmentSource.open(tmp_path),
        analysis={
            "capabilities": [capability],
            "tasks": [{"id": "solve", "description": "Solve the task"}],
        },
        version="1.2.3",
        local_image_ref="acme/repo:1.2.3",
        env_vars={"ANTHROPIC_API_KEY": "secret"},
    )

    assert lock_data["version"] == "2.0"
    assert lock_data["images"] == {
        "local": "acme/repo:1.2.3",
        "full": None,
        "pushed": None,
    }
    assert lock_data["build"]["baseImage"] == "python:3.11"
    assert lock_data["build"]["sourceHash"]
    assert lock_data["build"]["sourceFiles"] == [
        "Dockerfile.hud",
        "controller/server.py",
    ]
    assert lock_data["environment"]["variables"]["required"] == [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]
    # v6 manifest sections
    assert lock_data["capabilities"] == [capability]
    assert lock_data["tasks"] == [{"id": "solve", "description": "Solve the task"}]
