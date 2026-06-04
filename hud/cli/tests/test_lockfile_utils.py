from __future__ import annotations

from hud.cli.utils.lockfile import build_lock_data


def test_build_lock_data_builds_shared_lock_shape(tmp_path) -> None:
    (tmp_path / "Dockerfile.hud").write_text(
        "FROM python:3.11\nENV OPENAI_API_KEY=\n",
        encoding="utf-8",
    )
    controller_dir = tmp_path / "controller"
    controller_dir.mkdir()
    (controller_dir / "server.py").write_text("print('ok')\n", encoding="utf-8")

    capability = {"name": "shell", "protocol": "ssh/2", "url": "ssh://host:22", "params": {}}
    lock_data = build_lock_data(
        source_dir=tmp_path,
        # v6 analysis: the env's capabilities + tasks (from Environment.to_dict()).
        analysis={
            "capabilities": [capability],
            "tasks": [{"id": "solve", "description": "Solve the task"}],
        },
        version="1.2.3",
        image_name="acme/repo",
        build_id="build-1",
        build_method="modal",
        full_image_ref="acme/repo:1.2.3@sha256:abc",
        env_vars={"ANTHROPIC_API_KEY": "secret"},
        hud_version_value="modal-native",
    )

    assert lock_data["version"] == "2.0"
    assert lock_data["images"] == {
        "local": "acme/repo:1.2.3",
        "full": "acme/repo:1.2.3@sha256:abc",
        "pushed": None,
    }
    assert lock_data["build"]["buildId"] == "build-1"
    assert lock_data["build"]["buildMethod"] == "modal"
    assert lock_data["build"]["hudVersion"] == "modal-native"
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
