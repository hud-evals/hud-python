"""``hud deploy``: the build context it uploads, the build it triggers, and how it reports it."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.harness import Reply, write_files

from .conftest import BROWSER_PROJECT_ID, LOCKED_PROJECT_ID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from starlette.websockets import WebSocket

    from tests.harness import FakeServices, Hud, HudEnv, Request

REGISTRY_ID = "bbbbbbbb-0000-4000-8000-000000000001"
FIXED = {"source", "build_id", "name", "no_cache"}
SUCCEEDED = {
    "status": "SUCCEEDED",
    "version": "3",
    "uri": "registry.hud.test/example:3",
    "manifest": {"tasks": [{"id": "solve"}], "capabilities": [{"name": "shell"}]},
}
LOG_FRAMES = [
    {"type": "status", "message": "Build queued"},
    {"type": "log", "message": "Step 1/2 : FROM python:3.12\n", "timestamp": 0},
    {"type": "status_update", "status": "SUCCEEDED"},
    {"type": "complete", "message": "Build SUCCEEDED"},
]


def log_stream(
    frames: list[dict[str, Any]], *, close: tuple[int, str] | None = None
) -> Callable[[WebSocket, dict[str, str]], Awaitable[None]]:
    """A build-log socket that sends ``frames`` then closes, with ``close`` when given."""

    async def serve(websocket: WebSocket, params: dict[str, str]) -> None:
        del params
        await websocket.accept()
        for frame in frames:
            await websocket.send_text(json.dumps(frame))
        code, reason = close or (1000, "")
        await websocket.close(code=code, reason=reason)

    return serve


async def refuse(websocket: WebSocket, params: dict[str, str]) -> None:
    """A build-log socket that rejects the handshake."""
    del params
    await websocket.close(code=4404)


def trigger(request: Request) -> Reply:
    return Reply(
        json={"id": "build-1", "registry_id": request.json.get("registry_id", REGISTRY_ID)}
    )


@pytest.fixture
def builds(projects: FakeServices, hud_env: HudEnv, tmp_path: Path) -> FakeServices:
    """A platform that accepts one build, ``build-1``, and reports it SUCCEEDED.

    The presigned upload URL points back at the fake services; ``TMPDIR`` is
    private so a test can see that the build context tarball is removed.
    """
    (tmp_path / "tmp").mkdir()
    hud_env.set(TMPDIR=str(tmp_path / "tmp"), TZ="UTC")
    upload = {"upload_url": f"{projects.url('api')}/upload/build-1", "build_id": "build-1"}
    projects.route("api", "POST", "/v2/builds/upload-url", json=upload)
    projects.route("api", "PUT", "/upload/build-1", json={})
    projects.route("api", "POST", "/v2/builds/trigger", handler=trigger)
    projects.route("api", "GET", "/v2/builds/build-1/status", json=SUCCEEDED)
    projects.websocket("api", "/v2/builds/build-1/logs", log_stream(LOG_FRAMES))
    return projects


def environment(directory: Path, name: str = "example") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "env.py").write_text(f'from hud import Environment\nenv = Environment("{name}")\n')
    (directory / "Dockerfile.hud").write_text("FROM python:3.12\n")
    return directory


def uploaded(services: FakeServices) -> list[str]:
    """Member names of the build context tarball the platform received."""
    (put,) = services.requests("api", "PUT", "/upload/build-1")
    with tarfile.open(fileobj=io.BytesIO(put.body), mode="r:gz") as archive:
        return sorted(archive.getnames())


def trigger_body(services: FakeServices) -> dict[str, Any]:
    (body,) = services.bodies("api", "POST", "/v2/builds/trigger")
    return body


def test_deploy_uploads_the_context_triggers_the_build_and_links_the_directory(
    hud: Hud, builds: FakeServices, tmp_path: Path
) -> None:
    env = environment(hud.cwd / "env")
    write_files(
        env,
        {
            ".env": "SECRET=dotenv-secret\n",
            "token.txt": "file-secret",
            "vars.env": "KEY=file_value\n",
        },
    )

    result = hud(
        "deploy",
        "env",
        *("--env-file", "env/vars.env", "--build-arg", "PYTHON=3.12"),
        *(
            "--secret",
            "id=GITHUB_TOKEN,env=GITHUB_TOKEN",
            "--secret",
            "id=FILE_TOKEN,src=token.txt",
        ),
        env={"GITHUB_TOKEN": "env-secret"},
    )

    assert result.exit_code == 0, result
    assert "Step 1/2 : FROM python:3.12" in result.stderr  # the build log streams
    sent = trigger_body(builds)
    assert (sent["name"], sent["environment_variables"], sent["build_args"]) == (
        "example",
        {"KEY": "file_value"},
        {"PYTHON": "3.12"},
    )
    assert sent["build_secrets"] == {"GITHUB_TOKEN": "env-secret", "FILE_TOKEN": "file-secret"}
    assert ".env" not in uploaded(builds)
    assert not any(
        secret in str(result) for secret in ("dotenv-secret", "env-secret", "file-secret")
    )
    assert list((tmp_path / "tmp").iterdir()) == []
    assert json.loads((env / ".hud" / "config.json").read_text())["registry_id"] == REGISTRY_ID


def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    hud: Hud, builds: FakeServices
) -> None:
    environment(hud.cwd)
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

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == 0, result
    sent = uploaded(builds)
    assert {"keep.pyc", "src/build/out.o"} <= set(sent)
    assert not {"drop.pyc", "build/out.o", ".git/config", "nested/.env.prod"} & set(sent)


@pytest.mark.parametrize(
    ("files", "argv", "name"),
    [
        pytest.param({"env.py": 'env = Environment("my-env")\n'}, [], "my-env", id="declared"),
        pytest.param(
            {"env.py": 'actor = Environment("workspace")\nverifier = Environment("judge")\n'},
            ["--name", "judge"],
            "judge",
            id="chosen-among-several",
        ),
    ],
)
def test_the_environment_name_comes_from_its_declaration(
    hud: Hud, projects: FakeServices, files: dict[str, str], argv: list[str], name: str
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", *argv, "--dry-run", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert result.json["name"] == name


def test_a_dry_run_plans_without_prompting_writing_or_posting(
    hud: Hud, projects: FakeServices, hud_env: HudEnv
) -> None:
    environment(hud.cwd)
    write_files(hud.cwd, {".env": "SECRET=hidden", "vars.env": "KEY1=v\n"})
    before = sorted(path.name for path in hud.cwd.rglob("*"))

    result = hud(
        "deploy",
        *("--runtime", "MODAL", "--build-arg", "A=1", "--env-file", "vars.env"),
        *("--dry-run", "--json"),
    )

    assert result.exit_code == 0, result
    assert {key: result.json[key] for key in ("dry_run", "runtime", "env_var_keys")} == {
        "dry_run": True,
        "runtime": "modal",
        "env_var_keys": ["KEY1"],
    }
    assert sorted(path.name for path in hud.cwd.rglob("*")) == before
    assert not (hud_env.home / ".hud").exists()
    assert projects.requests("api", "POST") == []


def test_a_compose_runtime_config_reaches_the_build_as_its_document(
    hud: Hud, builds: FakeServices
) -> None:
    environment(hud.cwd)
    write_files(
        hud.cwd,
        {
            "project/compose.json": '{"services":{"main":{"image":"postgres:16"}}}',
            "runtime.json": json.dumps(
                {"compose": {"document": "project/compose.json", "root": "."}}
            ),
        },
    )

    result = hud("deploy", "--no-env", "--runtime-config", "runtime.json", "--json")

    compose = trigger_body(builds)["runtime_config"]["compose"]
    assert result.exit_code == 0, result
    assert compose["document"]["services"]["main"]["image"] == "postgres:16"
    assert compose["root"] == {"compose_path": "project/compose.json"}


@pytest.mark.parametrize(
    ("second", "body", "kept"),
    [
        ([], {}, True),
        (["--env-file", ".env"], {"environment_variables": {"SECRET": "dotenv"}}, False),
    ],
    ids=["again", "new-secrets"],
)
def test_a_redeploy_keeps_the_link_and_the_registry_secrets(
    hud: Hud, builds: FakeServices, second: list[str], body: dict[str, Any], kept: bool
) -> None:
    environment(hud.cwd)
    (hud.cwd / ".env").write_text("SECRET=dotenv")

    first = hud("deploy", "--env-file", ".env", "--json")
    linked = (hud.cwd / ".hud" / "config.json").read_bytes()
    again = hud("deploy", *second, "--json")

    assert (first.exit_code, again.exit_code) == (0, 0), again
    assert (hud.cwd / ".hud" / "config.json").read_bytes() == linked
    _, second_body = builds.bodies("api", "POST", "/v2/builds/trigger")
    assert {key: second_body[key] for key in second_body if key not in FIXED} == body
    assert ("Registry secrets: kept" in again.stderr) is kept


def test_a_first_deploy_lands_in_the_selected_project(
    hud: Hud, builds: FakeServices, hud_env: HudEnv
) -> None:
    environment(hud.cwd)
    hud_env.set(HUD_DEFAULT_PROJECT="browser-evals")

    result = hud("deploy", "--json")

    assert result.exit_code == 0, result
    assert trigger_body(builds)["project_id"] == BROWSER_PROJECT_ID


@pytest.mark.parametrize(
    ("files", "argv", "exit_code", "error"),
    [
        pytest.param({}, [], 2, "usage", id="no-environment-declared"),
        pytest.param(
            {"a.py": 'a = Environment("a")\n', "b.py": 'b = Environment("b")\n'},
            [],
            2,
            "usage",
            id="several-environments-and-no-name",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--project", LOCKED_PROJECT_ID],
            1,
            "permission_denied",
            id="a-read-only-project",
        ),
    ],
)
def test_deploy_refuses_before_uploading(
    hud: Hud,
    projects: FakeServices,
    files: dict[str, str],
    argv: list[str],
    exit_code: int,
    error: str,
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result
    assert projects.requests("api", "POST") == []


def test_a_rejected_build_request_leaves_nothing_behind(
    hud: Hud, builds: FakeServices, tmp_path: Path
) -> None:
    builds.route("api", "POST", "/v2/builds/trigger", Reply(status=409, json={"detail": "Busy"}))
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "conflict"), result
    assert list((tmp_path / "tmp").iterdir()) == []
    assert not (hud.cwd / ".hud").exists()


@pytest.mark.parametrize(
    ("status", "stream", "exit_code"),
    [
        pytest.param(
            {"status": "FAILED", "error_message": "pip install failed"},
            log_stream([{"type": "error", "error": "Step 2 failed"}]),
            1,
            id="failed-build",
        ),
        pytest.param(SUCCEEDED, refuse, 0, id="no-stream"),
    ],
)
def test_the_build_status_decides_the_exit_whatever_the_log_stream_does(
    hud: Hud,
    builds: FakeServices,
    status: dict[str, Any],
    stream: Callable[[WebSocket, dict[str, str]], Awaitable[None]],
    exit_code: int,
) -> None:
    builds.route("api", "GET", "/v2/builds/build-1/status", json=status)
    builds.websocket("api", "/v2/builds/build-1/logs", stream)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == exit_code, result
    assert (result.json["status"], result.json["success"]) == (status["status"], exit_code == 0)
