"""``hud deploy``: the build context it uploads, the build it triggers, and how it reports it."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.harness import Reply

from .conftest import BROWSER_PROJECT_ID, LOCKED_PROJECT_ID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from starlette.websockets import WebSocket

    from tests.harness import FakeServices, Hud, HudEnv, Request

REGISTRY_ID = "bbbbbbbb-0000-4000-8000-000000000001"
OTHER_REGISTRY_ID = "bbbbbbbb-0000-4000-8000-000000000002"
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


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        if name.endswith("/"):
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


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
    write(
        env,
        {
            "server/pkg/tool.py": "print('tool')\n",
            ".env": "SECRET=dotenv-secret\n",
            "token.txt": "file-secret",
            "runtime.json": json.dumps({"limits": {"startup_timeout_s": 300}}),
            "vars.env": "# comment\nKEY1=file_value\nEMPTY=\nNOEQ\nKEY2=file_value\n",
        },
    )

    result = hud(
        "deploy",
        "env",
        *("--env-file", "env/vars.env", "--env", "KEY1=flag_value"),
        *("--build-arg", "PYTHON=3.12"),
        *(
            "--secret",
            "id=GITHUB_TOKEN,env=GITHUB_TOKEN",
            "--secret",
            "id=FILE_TOKEN,src=token.txt",
        ),
        *("--runtime", "HUD", "--runtime-config", "env/runtime.json", "--no-cache", "--json"),
        env={"GITHUB_TOKEN": "env-secret"},
    )

    assert result.exit_code == 0, result
    assert (result.json["success"], result.json["build_id"], result.json["status"]) == (
        True,
        "build-1",
        "SUCCEEDED",
    )
    assert trigger_body(builds) == {
        "source": "direct",
        "build_id": "build-1",
        "name": "example",
        "no_cache": True,
        "runtime_provider": "hud",
        "runtime_config": {"limits": {"startup_timeout_s": 300}},
        "environment_variables": {"KEY1": "flag_value", "EMPTY": "", "KEY2": "file_value"},
        "build_args": {"PYTHON": "3.12"},
        "build_secrets": {"GITHUB_TOKEN": "env-secret", "FILE_TOKEN": "file-secret"},
    }
    assert ".env" not in uploaded(builds) and "server/pkg/tool.py" in uploaded(builds)
    assert not any(
        secret in str(result) for secret in ("dotenv-secret", "env-secret", "file-secret")
    )
    assert list((tmp_path / "tmp").iterdir()) == []
    assert json.loads((env / ".hud" / "config.json").read_text())["registry_id"] == REGISTRY_ID


def test_deploy_streams_the_build_log_to_the_terminal(hud: Hud, builds: FakeServices) -> None:
    environment(hud.cwd)

    result = hud("deploy", "--no-env")

    assert result.exit_code == 0, result
    assert "Step 1/2 : FROM python:3.12" in result.stderr
    assert "https://hud.example/environments/" in result.stderr


def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    hud: Hud, builds: FakeServices
) -> None:
    environment(hud.cwd)
    write(
        hud.cwd,
        {
            "main.py": "",
            ".env": "SECRET=1",
            ".env.prod": "SECRET=1",
            "nested/.env": "SECRET=1",
            ".git/config": "",
            "node_modules/dep.js": "",
            "__pycache__/x.pyc": "",
            "build/out.o": "",
            "src/build/out.o": "",
            "keep.pyc": "",
            "drop.pyc": "",
            "empty/": "",
            ".dockerignore": "*.pyc\n!keep.pyc\n/build\n!node_modules\n",
        },
    )

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert uploaded(builds) == [
        ".dockerignore",
        "Dockerfile.hud",
        "empty",
        "env.py",
        "keep.pyc",
        "main.py",
        "nested",
        "node_modules",
        "node_modules/dep.js",
        "src",
        "src/build",
        "src/build/out.o",
    ]


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
        pytest.param(
            {
                "server/pkg/env.py": 'env = Environment("nested")\n',
                ".venv/env.py": 'env = Environment("excluded")\n',
                "broken.py": "def broken(:\n",
            },
            [],
            "nested",
            id="found-past-virtualenvs-and-broken-files",
        ),
    ],
)
def test_the_environment_name_comes_from_its_declaration(
    hud: Hud, projects: FakeServices, files: dict[str, str], argv: list[str], name: str
) -> None:
    write(hud.cwd, files)

    result = hud("deploy", *argv, "--dry-run", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert result.json["name"] == name


def test_a_dry_run_plans_without_prompting_writing_or_posting(
    hud: Hud, projects: FakeServices, hud_env: HudEnv
) -> None:
    environment(hud.cwd)
    write(hud.cwd, {".env": "SECRET=hidden", "vars.env": "KEY1=v\n"})
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
    write(
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


def test_a_plain_deploy_sends_only_the_build(hud: Hud, builds: FakeServices) -> None:
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert trigger_body(builds) == {
        "source": "direct",
        "build_id": "build-1",
        "name": "example",
        "no_cache": False,
    }


@pytest.mark.parametrize(
    ("second", "body", "kept"),
    [
        ([], {}, True),
        (["--registry-id", OTHER_REGISTRY_ID], {"registry_id": OTHER_REGISTRY_ID}, True),
        (["--project", BROWSER_PROJECT_ID], {"project_id": BROWSER_PROJECT_ID}, True),
        (["--env-file", ".env"], {"environment_variables": {"SECRET": "dotenv"}}, False),
    ],
    ids=["again", "another-registry", "another-project", "new-secrets"],
)
def test_a_redeploy_keeps_the_link_and_the_registry_secrets(
    hud: Hud, builds: FakeServices, second: list[str], body: dict[str, Any], kept: bool
) -> None:
    builds.route(
        "api",
        "GET",
        f"/v2/registry/{OTHER_REGISTRY_ID}",
        json={"id": OTHER_REGISTRY_ID, "name": "example"},
    )
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


@pytest.mark.parametrize(
    ("argv", "linked", "machine_default", "body"),
    [
        pytest.param(
            ["--project", "browser-evals"],
            None,
            "default",
            {"project_id": BROWSER_PROJECT_ID},
            id="the-flag-beats-the-machine-default",
        ),
        pytest.param(
            [],
            "browser-evals",
            "locked-down",
            {"project_id": BROWSER_PROJECT_ID},
            id="the-directory-link-beats-the-machine-default",
        ),
        pytest.param(
            [], None, "browser-evals", {"project_id": BROWSER_PROJECT_ID}, id="the-machine-default"
        ),
        pytest.param([], None, None, {}, id="the-team-default"),
    ],
)
def test_a_first_deploy_lands_in_the_project_its_selection_order_picks(
    hud: Hud,
    builds: FakeServices,
    hud_env: HudEnv,
    argv: list[str],
    linked: str | None,
    machine_default: str | None,
    body: dict[str, Any],
) -> None:
    environment(hud.cwd)
    if linked is not None:
        assert hud("project", "use", linked).exit_code == 0
    hud_env.set(HUD_DEFAULT_PROJECT=machine_default)

    result = hud("deploy", *argv, "--json")

    assert result.exit_code == 0, result
    assert {key: value for key, value in trigger_body(builds).items() if key not in FIXED} == body


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
            {"env.py": 'env = Environment("My Env")\n'},
            ["--registry-id", REGISTRY_ID],
            2,
            "usage",
            id="a-registry-id-naming-another-environment",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--env", "INVALID"],
            2,
            "usage",
            id="a-malformed-flag",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--env-file", "missing.env"],
            1,
            "not_found",
            id="a-missing-env-file",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n', ".env": "SECRET=1"},
            [],
            2,
            "usage",
            id="a-dotenv-and-no-choice-about-it",
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
    projects.route(
        "api", "GET", f"/v2/registry/{REGISTRY_ID}", json={"id": REGISTRY_ID, "name": "other-env"}
    )
    write(hud.cwd, files)

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
        pytest.param(
            SUCCEEDED, log_stream([], close=(4003, "token expired")), 0, id="access-denied"
        ),
        pytest.param(SUCCEEDED, log_stream([], close=(1011, "restarting")), 0, id="stream-closed"),
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
