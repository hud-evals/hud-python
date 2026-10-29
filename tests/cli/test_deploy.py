"""``hud deploy``: the build context it uploads, what rides the build trigger instead, and
how the build's end decides the exit."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
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


HARBOR_TASK = {
    "hello-world/task.toml": 'version = "1.0"\n',
    "hello-world/instruction.md": "Say hello.\n",
    "hello-world/environment/Dockerfile": "FROM ubuntu:24.04\n",
    "hello-world/environment/.env": "GREETING=hello\n",
    "hello-world/tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
}
IMPORT_ID = "66666666-6666-4666-8666-666666666666"
FILE_ID = "ffffffff-0000-4000-8000-000000000001"
TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
TASK_ID = "dddddddd-0000-4000-8000-000000000001"
REGISTERED = {
    "name": "hello-world",
    "source": "bundle",
    "path": "hello-world",
    "environment_id": ENV_ID,
    "build_id": BUILD_ID,
    "build_version": 1,
    "task_id": TASK_ID,
    "status": "imported",
}


def harbor_report(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "harbor_import_report.v5",
        "taskset_id": TASKSET_ID,
        "tasks": [task],
        "jobs": [],
        "warnings": [],
        "error": None,
    }


@pytest.fixture
def harbor(builds: FakeServices, hud: Hud) -> FakeServices:
    """A Harbor task folder, and a platform that imports it into a new ``tb2`` taskset as the
    unbuilt version ``BUILD_ID`` of ``ENV_ID``, then builds that version."""
    write_files(hud.cwd, HARBOR_TASK)
    formats = [{"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}]
    builds.route("api", "GET", "/v2/imports/formats", json={"formats": formats})
    builds.route(
        "api",
        "POST",
        "/v2/imports/uploads",
        json={
            "file": {"id": FILE_ID, "filename": "hello-world.zip"},
            "upload_url": f"{builds.url('api')}/uploads/{FILE_ID}",
        },
    )
    builds.route("api", "POST", "/v2/data/{id}/complete", json={"id": FILE_ID})
    builds.route(
        "api", "GET", "/v2/tasksets/by-name/{name}", status=404, json={"detail": "Not found"}
    )
    builds.route(
        "api", "POST", "/v2/tasksets", json={"id": TASKSET_ID, "name": "tb2", "can_edit": True}
    )
    builds.route(
        "api",
        "POST",
        "/v2/imports",
        json={
            "id": IMPORT_ID,
            "format": "harbor",
            "taskset_id": TASKSET_ID,
            "file": {"id": FILE_ID, "filename": "hello-world.zip", "size_bytes": 1},
            "status": "completed",
            "summary": "Imported 1 task(s)",
            "created_at": "2026-10-07T12:00:00Z",
            "completed_at": "2026-10-07T12:01:00Z",
        },
    )
    builds.route("api", "GET", f"/v2/imports/{IMPORT_ID}/report", json=harbor_report(REGISTERED))
    builds.route(
        "api",
        "GET",
        f"/v2/builds/{BUILD_ID}/status",
        Reply(json={"status": "UPLOADED"}),
        Reply(json=SUCCEEDED),
    )
    return builds


def imported_bundle(services: FakeServices) -> set[str]:
    """Member names of the Harbor bundle the platform received."""
    (upload,) = services.requests("api", "PUT", "/uploads/{id}")
    with zipfile.ZipFile(io.BytesIO(upload.body)) as archive:
        return set(archive.namelist())


def test_a_harbor_task_folder_is_imported_into_its_taskset_and_the_version_it_holds_is_built(
    harbor: FakeServices, hud: Hud
) -> None:
    result = hud(
        *("deploy", "hello-world", "--taskset", "tb2", "--env", "TOKEN=t"),
        *("--yes", "--json"),
    )

    assert result.exit_code == 0, result
    assert imported_bundle(harbor) == set(HARBOR_TASK)
    assert harbor.bodies("api", "POST", "/v2/imports")[0]["taskset_id"] == TASKSET_ID
    assert harbor.requests("api", "POST", "/v2/builds/register") == []
    assert triggered(harbor) == {
        "source": "direct",
        "registry_id": ENV_ID,
        "build_id": BUILD_ID,
        "no_cache": False,
        "environment_variables": {"TOKEN": "t"},
    }
    assert {key: result.json[key] for key in ("status", "taskset_id", "task_id", "import_id")} == {
        "status": "SUCCEEDED",
        "taskset_id": TASKSET_ID,
        "task_id": TASK_ID,
        "import_id": IMPORT_ID,
    }
    assert not (hud.cwd / "hello-world" / ".hud").exists()


def test_upload_only_adds_the_task_and_builds_nothing(harbor: FakeServices, hud: Hud) -> None:
    result = hud("deploy", "hello-world", "--upload-only", "--yes", "--json")

    assert result.exit_code == 0, result
    assert harbor.bodies("api", "POST", "/v2/tasksets")[0]["name"] == "hello-world"
    assert harbor.requests("api", "POST", "/v2/builds/trigger") == []
    assert (result.json["status"], result.json["task_id"]) == ("UPLOADED", TASK_ID)


@pytest.mark.parametrize(("status", "exit_code"), [("SUCCEEDED", 0), ("FAILED", 1)])
def test_a_version_whose_build_ended_is_reported_and_not_built_again(
    status: str, exit_code: int, harbor: FakeServices, hud: Hud
) -> None:
    harbor.route(
        "api", "GET", f"/v2/builds/{BUILD_ID}/status", json={**SUCCEEDED, "status": status}
    )

    result = hud("deploy", "hello-world", "--yes", "--json")

    assert (result.exit_code, result.json["status"]) == (exit_code, status), result
    assert harbor.requests("api", "POST", "/v2/builds/trigger") == []


def test_a_folder_the_import_does_not_register_builds_nothing(
    harbor: FakeServices, hud: Hud
) -> None:
    skipped = {
        "name": "hello-world",
        "source": "bundle",
        "path": "hello-world",
        "status": "skipped",
        "reason": "not a valid Harbor task: instruction.md is missing",
    }
    harbor.route("api", "GET", f"/v2/imports/{IMPORT_ID}/report", json=harbor_report(skipped))

    result = hud("deploy", "hello-world", "--yes", "--json")

    assert (result.exit_code, result.json["success"]) == (1, False), result
    assert "instruction.md is missing" in result.stderr
    assert harbor.requests("api", "GET", f"/v2/builds/{BUILD_ID}/status") == []
    assert harbor.requests("api", "POST", "/v2/builds/trigger") == []


@pytest.mark.parametrize(
    ("flags", "refused"),
    [
        (("--name", "greet"), "--name"),
        (("--registry-id", ENV_ID), "--registry-id"),
        (("--runtime-config", "runtime.json"), "--runtime-config"),
        (("--build-arg", "A=1"), "--build-arg"),
        (("--secret", "id=TOKEN,env=TOKEN"), "--secret"),
        (("--runtime", "hud"), "--runtime"),
        (("--upload-only", "--env", "A=1"), "--env"),
        (("--upload-only", "--no-cache"), "--no-cache"),
    ],
    ids=lambda value: " ".join(value) if isinstance(value, tuple) else "",
)
def test_a_harbor_task_folder_takes_no_build_configuration(
    flags: tuple[str, ...], refused: str, signed_in: FakeServices, hud: Hud
) -> None:
    write_files(hud.cwd, HARBOR_TASK)

    result = hud("deploy", "hello-world", *flags, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert refused in result.json["message"]
    assert signed_in.requests("api", "POST") == []


@pytest.mark.parametrize("flags", [("--taskset", "tb2"), ("--upload-only",)], ids=str)
def test_an_environment_deploy_takes_no_harbor_flags(
    flags: tuple[str, ...], builds: FakeServices, hud: Hud
) -> None:
    result = hud("deploy", *flags, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert builds.requests("api", "POST") == []


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
