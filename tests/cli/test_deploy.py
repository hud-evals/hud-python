"""``hud deploy``: the build context it uploads, the build it triggers with what its options
name, where the environment lands, and how the build's end decides the exit."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import write_files

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.websockets import WebSocket

    from tests.conftest import Hud, HudEnv
    from tests.fixtures.services import FakeServices

ENV_PY = 'from hud import Environment\nenv = Environment("example")\n'
BUILD_ID = "build-1"
ENV_ID = "bbbbbbbb-0000-4000-8000-000000000042"
PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000002"
SUCCEEDED = {
    "status": "SUCCEEDED",
    "version": "1",
    "uri": "registry.hud.test/env:1",
    "manifest": {"tasks": [], "capabilities": []},
}
LOG = [
    {"type": "log", "message": "Step 1/2 : FROM python:3.12\n", "timestamp": 0},
    {"type": "status_update", "status": "SUCCEEDED"},
    {"type": "complete", "message": "Build SUCCEEDED"},
]


def streaming(frames: list[dict[str, Any]]) -> Any:
    """A build-log socket that sends ``frames`` and closes."""

    async def handler(websocket: WebSocket, params: dict[str, str]) -> None:
        await websocket.accept()
        for frame in frames:
            await websocket.send_text(json.dumps(frame))
        await websocket.close()

    return handler


@pytest.fixture
def builds(signed_in: FakeServices, hud: Hud) -> FakeServices:
    """A directory holding the ``example`` environment, and a platform that builds it."""
    write_files(hud.cwd, {"env.py": ENV_PY, "Dockerfile.hud": "FROM python:3.12\n"})
    upload = f"{signed_in.url('api')}/uploads/{BUILD_ID}"
    signed_in.route(
        "api", "POST", "/v2/builds/upload-url", json={"upload_url": upload, "build_id": BUILD_ID}
    )
    signed_in.route("api", "PUT", "/uploads/{id}", json={})
    signed_in.route(
        "api", "POST", "/v2/builds/trigger", json={"id": BUILD_ID, "registry_id": ENV_ID}
    )
    signed_in.route("api", "GET", f"/v2/builds/{BUILD_ID}/status", json=SUCCEEDED)
    signed_in.websocket("api", "/v2/builds/{id}/logs", streaming(LOG))
    return signed_in


def uploaded(services: FakeServices) -> set[str]:
    """Member names of the build context tarball the platform received."""
    (upload,) = services.requests("api", "PUT", "/uploads/{id}")
    with tarfile.open(fileobj=io.BytesIO(upload.body), mode="r:gz") as archive:
        return set(archive.getnames())


def triggered(services: FakeServices) -> dict[str, Any]:
    (trigger,) = services.bodies("api", "POST", "/v2/builds/trigger")
    return trigger


def test_env_files_build_args_and_secrets_ride_the_trigger_and_never_the_context(
    builds: FakeServices, hud: Hud
) -> None:
    write_files(
        hud.cwd, {".env": "SECRET=dotenv\n", "token.txt": "file-secret", "vars.env": "KEY=v\n"}
    )

    result = hud(
        *("deploy", "--env-file", "vars.env", "--build-arg", "PYTHON=3.12"),
        *("--secret", "id=GITHUB_TOKEN,env=GITHUB_TOKEN"),
        *("--secret", "id=FILE_TOKEN,src=token.txt"),
        *("--yes", "--json"),
        env={"GITHUB_TOKEN": "env-secret"},
    )

    assert result.exit_code == 0, result
    trigger = triggered(builds)
    assert (trigger["name"], trigger["build_id"]) == ("example", BUILD_ID)
    assert (trigger["environment_variables"], trigger["build_args"], trigger["build_secrets"]) == (
        {"KEY": "v"},
        {"PYTHON": "3.12"},
        {"GITHUB_TOKEN": "env-secret", "FILE_TOKEN": "file-secret"},
    )
    assert ".env" not in uploaded(builds)


def test_a_compose_runtime_config_reaches_the_build_as_its_document(
    builds: FakeServices, hud: Hud
) -> None:
    write_files(
        hud.cwd,
        {
            "project/compose.json": '{"services":{"main":{"image":"postgres:16"}}}',
            "runtime.json": json.dumps(
                {"compose": {"document": "project/compose.json", "root": "."}}
            ),
        },
    )

    result = hud("deploy", "--no-env", "--runtime-config", "runtime.json", "--yes", "--json")

    assert result.exit_code == 0, result
    compose = triggered(builds)["runtime_config"]["compose"]
    assert compose["document"]["services"]["main"]["image"] == "postgres:16"
    assert compose["root"] == {"compose_path": "project/compose.json"}


def test_a_first_deploy_lands_in_the_selected_project_and_links_the_directory(
    builds: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    project = {"id": PROJECT_ID, "name": "browser-evals", "capabilities": {"create": True}}
    builds.route("api", "GET", "/v2/projects", json={"items": [project], "total": 1})
    hud_env.set(HUD_DEFAULT_PROJECT="browser-evals")

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == 0, result
    assert triggered(builds)["project_id"] == PROJECT_ID
    assert json.loads((hud.cwd / ".hud" / "config.json").read_text())["registry_id"] == ENV_ID


def test_a_redeploy_keeps_the_registrys_variables_unless_given_new_ones(
    builds: FakeServices, hud: Hud
) -> None:
    write_files(hud.cwd, {".env": "SECRET=dotenv\n"})

    first = hud("deploy", "--env-file", ".env", "--yes", "--json")
    second = hud("deploy", "--yes", "--json")

    assert (first.exit_code, second.exit_code) == (0, 0), second
    assert [
        "environment_variables" in body
        for body in builds.bodies("api", "POST", "/v2/builds/trigger")
    ] == [
        True,
        False,
    ]


def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    builds: FakeServices, hud: Hud
) -> None:
    write_files(
        hud.cwd,
        {
            "nested/.env.prod": "SECRET=1",
            ".git/config": "",
            "build/out.o": "",
            "src/build/out.o": "",
            "keep.pyc": "",
            "drop.pyc": "",
            ".dockerignore": "*.pyc\n!keep.pyc\n/build\n",
        },
    )

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == 0, result
    sent = uploaded(builds)
    assert {"keep.pyc", "src/build/out.o"} <= sent
    assert not {"drop.pyc", "build/out.o", ".git/config", "nested/.env.prod"} & sent


@pytest.mark.parametrize(
    ("files", "argv", "planned"),
    [
        ({"env.py": 'env = Environment("my-env")\n'}, [], {"name": "my-env"}),
        (
            {"env.py": 'actor = Environment("workspace")\nverifier = Environment("judge")\n'},
            ["--name", "judge"],
            {"name": "judge"},
        ),
        ({".env": "SECRET=hidden"}, [], {"name": "example", "dotenv_pending": True}),
    ],
    ids=["declared", "chosen-among-several", "a-dotenv-beside-it-is-asked-about"],
)
def test_a_dry_run_plans_the_environment_it_would_deploy(
    files: dict[str, str], argv: list[str], planned: dict[str, Any], builds: FakeServices, hud: Hud
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert {key: result.json[key] for key in planned} == planned
    assert builds.requests("api", "POST") == []


@pytest.mark.parametrize(
    ("status", "frames", "exit_code", "logged"),
    [
        (SUCCEEDED, LOG, 0, "Step 1/2 : FROM python:3.12"),
        (
            {"status": "FAILED", "error_message": "pip install failed"},
            [{"type": "error", "error": "Step 2 failed"}],
            1,
            "Build error: Step 2 failed",
        ),
    ],
    ids=["a-build-that-succeeds-streams-its-log", "a-build-that-fails"],
)
def test_the_build_decides_the_exit_and_no_context_is_left_behind(
    status: dict[str, Any],
    frames: list[dict[str, Any]],
    exit_code: int,
    logged: str,
    builds: FakeServices,
    hud_env: HudEnv,
    hud: Hud,
    tmp_path: Path,
) -> None:
    (tmp_path / "tmp").mkdir()
    hud_env.set(TMPDIR=str(tmp_path / "tmp"))
    builds.route("api", "GET", f"/v2/builds/{BUILD_ID}/status", json=status)
    builds.websocket("api", "/v2/builds/{id}/logs", streaming(frames))

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert (result.exit_code, result.json["success"]) == (exit_code, exit_code == 0), result
    assert logged in result.stderr
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_build_request_the_platform_rejects_leaves_no_context_behind(
    builds: FakeServices, hud_env: HudEnv, hud: Hud, tmp_path: Path
) -> None:
    (tmp_path / "tmp").mkdir()
    hud_env.set(TMPDIR=str(tmp_path / "tmp"))
    builds.route("api", "POST", "/v2/builds/trigger", status=409, json={"detail": "Busy"})

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "conflict"), result
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize(
    "files",
    [{}, {"a.py": 'a = Environment("a")\n', "b.py": 'b = Environment("b")\n'}],
    ids=["no-environment-declared", "several-environments-and-no-name"],
)
def test_a_deploy_that_cannot_tell_what_to_build_is_a_usage_error(
    files: dict[str, str], signed_in: FakeServices, hud: Hud
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert signed_in.requests("api", "POST") == []
