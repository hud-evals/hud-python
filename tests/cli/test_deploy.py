"""``hud deploy``: the build context it uploads, what rides the build trigger instead, and
how the build's end decides the exit."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import write_files
from tests.fixtures.services import Reply

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
    ("status", "frames", "exit_code", "logged"),
    [
        (SUCCEEDED, LOG, 0, "Step 1/2 : FROM python:3.12"),
        (
            {"status": "FAILED", "error_message": "pip install failed"},
            [{"type": "error", "error": "Step 2 failed"}],
            1,
            "Step 2 failed",
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


TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
IMPORT_ID = "66666666-6666-4666-8666-666666666666"


@pytest.fixture
def harbor(builds: FakeServices, hud: Hud) -> FakeServices:
    """A Harbor task folder, and a platform whose import registers it as the unbuilt build
    ``BUILD_ID`` of ``ENV_ID``."""
    write_files(hud.cwd, {"hello-world/task.toml": 'version = "1.0"\n'})
    formats = [{"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}]
    builds.route("api", "GET", "/v2/imports/formats", json={"formats": formats})
    builds.route(
        "api",
        "POST",
        "/v2/imports/uploads",
        json={"file": {"id": "file-1"}, "upload_url": f"{builds.url('api')}/uploads/file-1"},
    )
    builds.route("api", "POST", "/v2/data/{id}/complete", json={})
    builds.route("api", "GET", "/v2/tasksets/by-name/{name}", status=404, json={})
    builds.route("api", "POST", "/v2/tasksets", json={"id": TASKSET_ID, "name": "tb2"})
    builds.route(
        "api",
        "POST",
        "/v2/imports",
        json={
            "id": IMPORT_ID,
            "format": "harbor",
            "taskset_id": TASKSET_ID,
            "file": {"id": "file-1", "filename": "hello-world.zip", "size_bytes": 1},
            "status": "completed",
            "created_at": "2026-10-07T12:00:00Z",
        },
    )
    task = {
        "name": "hello-world",
        "source": "bundle",
        "path": "hello-world",
        "environment_id": ENV_ID,
        "build_id": BUILD_ID,
        "status": "imported",
    }
    builds.route(
        "api",
        "GET",
        f"/v2/imports/{IMPORT_ID}/report",
        json={
            "schema_version": "harbor_import_report.v5",
            "taskset_id": TASKSET_ID,
            "tasks": [task],
            "jobs": [],
            "warnings": [],
        },
    )
    builds.route(
        "api",
        "GET",
        f"/v2/builds/{BUILD_ID}/status",
        Reply(json={"status": "UPLOADED"}),
        Reply(json=SUCCEEDED),
    )
    return builds


def test_a_harbor_task_folder_is_imported_and_the_build_it_registered_is_built(
    harbor: FakeServices, hud: Hud
) -> None:
    result = hud("deploy", "hello-world", "-e", "TOKEN=t", "--yes", "--json")

    assert result.exit_code == 0, result
    assert triggered(harbor) == {
        "source": "direct",
        "registry_id": ENV_ID,
        "build_id": BUILD_ID,
        "no_cache": False,
        "environment_variables": {"TOKEN": "t"},
    }


def test_upload_only_imports_the_folder_and_builds_nothing(harbor: FakeServices, hud: Hud) -> None:
    result = hud("deploy", "hello-world", "--upload-only", "--yes", "--json")

    assert result.exit_code == 0, result
    assert harbor.requests("api", "POST", "/v2/builds/trigger") == []


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
