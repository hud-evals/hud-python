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

    lock_data = build_lock_data(
        source_dir=tmp_path,
        analysis={
            "initializeMs": 123,
            "toolCount": 1,
            "internalToolCount": 1,
            "tools": [
                {
                    "name": "setup",
                    "description": "Calls internal functions.",
                    "inputSchema": {"type": "object"},
                    "internalTools": ["prepare"],
                }
            ],
            "prompts": [],
            "resources": [],
            "scenarios": [],
            "hubTools": {"setup": ["prepare"]},
        },
        version="1.2.3",
        image_name="acme/repo",
        build_id="build-1",
        build_method="modal",
        full_image_ref="acme/repo:1.2.3@sha256:abc",
        env_vars={"ANTHROPIC_API_KEY": "secret"},
        hud_version_value="modal-native",
    )

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
    assert lock_data["environment"]["initializeMs"] == 123
    assert lock_data["environment"]["toolCount"] == 1
    assert lock_data["environment"]["internalToolCount"] == 1
    assert lock_data["environment"]["variables"]["required"] == [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]
    assert lock_data["tools"] == [
        {
            "name": "setup",
            "description": "Calls internal functions.",
            "inputSchema": {"type": "object"},
            "internalTools": ["prepare"],
        }
    ]
    assert lock_data["hubTools"] == {"setup": ["prepare"]}
