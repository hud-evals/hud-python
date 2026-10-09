"""``hud deploy``: the build context upload, the build trigger, and the build log stream."""

from __future__ import annotations

import io
import json
import re
import tarfile
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

from tests.harness import Reply, scrub

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
    {"type": "status_update", "status": "IN_PROGRESS"},
    {"type": "log", "message": "Step 1/2 : FROM python:3.12\n", "timestamp": 0},
    {"type": "log", "message": ""},
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
    (env / "server" / "pkg").mkdir(parents=True)
    (env / "server" / "pkg" / "tool.py").write_text("print('tool')\n")
    (env / ".env").write_text("SECRET=dotenv-secret\n")
    (env / "token.txt").write_text("file-secret")
    (env / "runtime.json").write_text(json.dumps({"limits": {"startup_timeout_s": 300}}))
    (env / "vars.env").write_text("# comment\nKEY1=file_value\nEMPTY=\nNOEQ\nKEY2=file_value\n")

    result = hud(
        "deploy",
        "env",
        "--env-file",
        "env/vars.env",
        "--env",
        "KEY1=flag_value",
        "--build-arg",
        "PYTHON=3.12",
        "--secret",
        "id=GITHUB_TOKEN,env=GITHUB_TOKEN",
        "--secret",
        "id=FILE_TOKEN,src=token.txt",
        "--runtime",
        "HUD",
        "--runtime-config",
        "env/runtime.json",
        "--no-cache",
        "--json",
        env={"GITHUB_TOKEN": "env-secret"},
    )

    assert result.exit_code == 0, result
    assert result.json == snapshot(
        {
            "success": True,
            "action": "deploy",
            "build_id": "build-1",
            "registry_id": "bbbbbbbb-0000-4000-8000-000000000001",
            "status": "SUCCEEDED",
            "name": "example",
            "dry_run": False,
            "runtime": None,
            "env_var_keys": [],
            "build_arg_keys": [],
            "dotenv_pending": False,
            "details": {
                "status": "SUCCEEDED",
                "version": "3",
                "uri": "registry.hud.test/example:3",
                "manifest": {"tasks": [{"id": "solve"}], "capabilities": [{"name": "shell"}]},
            },
        }
    )
    assert trigger_body(builds) == snapshot(
        {
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
    )
    assert uploaded(builds) == snapshot(
        [
            "Dockerfile.hud",
            "env.py",
            "runtime.json",
            "server",
            "server/pkg",
            "server/pkg/tool.py",
            "token.txt",
        ]
    )
    output = result.stdout + result.stderr
    assert not any(secret in output for secret in ("dotenv-secret", "env-secret", "file-secret"))
    assert list((tmp_path / "tmp").iterdir()) == []
    assert json.loads((env / ".hud" / "config.json").read_text())["registry_id"] == REGISTRY_ID


def test_deploy_streams_the_build_log_and_summarizes_the_build(
    hud: Hud, builds: FakeServices
) -> None:
    environment(hud.cwd)

    result = hud("deploy", "--no-env")

    assert result.exit_code == 0, result
    assert result.stderr[result.stderr.index("Build Logs") :] == snapshot("""\
Build Logs
Connecting to build logs stream...
Build queued
[00:00:00] Step 1/2 : FROM python:3.12
Build status: SUCCEEDED
Build SUCCEEDED

Build
 Environment   example                     \n\
 Status        SUCCEEDED                   \n\
 Version       3                           \n\
 Image         registry.hud.test/example:3 \n\
 Tasks         solve                       \n\
 Capabilities  shell                       \n\
https://hud.example/environments/bbbbbbbb-0000-4000-8000-000000000001
""")
    (socket,) = builds.requests("api", "WS")
    assert socket.query == {"api_key": ["sk-hud-test"]}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {
                "main.py": "",
                ".env": "SECRET=1",
                ".env.prod": "SECRET=1",
                "service.env": "SECRET=1",
                ".git/config": "",
                "nested/.env": "SECRET=1",
                "keep.pyc": "",
                ".dockerignore": "!*\n",
            },
            [".dockerignore", "Dockerfile.hud", "env.py", "keep.pyc", "main.py", "nested"],
            id="secrets-are-never-uploaded",
        ),
        pytest.param(
            {
                "packages/": "",
                "ignored/": "",
                ".gitignore": "data/\nbundle.bin\n",
                "bundle.bin": "",
                "data/fixture.json": "{}",
                ".dockerignore": "ignored/\n",
            },
            [
                ".dockerignore",
                ".gitignore",
                "Dockerfile.hud",
                "bundle.bin",
                "data",
                "data/fixture.json",
                "env.py",
                "packages",
            ],
            id="empty-dirs-kept-gitignore-ignored",
        ),
        pytest.param(
            {
                "dist/app.whl": "",
                "node_modules/dep.js": "",
                "__pycache__/x.pyc": "",
                ".venv/bin/python": "",
                ".dockerignore": "!node_modules\n",
            },
            [
                ".dockerignore",
                "Dockerfile.hud",
                "dist",
                "dist/app.whl",
                "env.py",
                "node_modules",
                "node_modules/dep.js",
            ],
            id="default-junk-can-be-reincluded",
        ),
        pytest.param(
            {
                "keep.pyc": "",
                "drop.pyc": "",
                "a.py": "",
                "pkg/a.pyc": "",
                "node_modules/index.js": "",
                ".dockerignore": "# comment\n\n*.pyc\n!keep.pyc\nnode_modules/\n",
            },
            [".dockerignore", "Dockerfile.hud", "a.py", "env.py", "keep.pyc", "pkg"],
            id="globs-last-match-and-directory-patterns",
        ),
        pytest.param(
            {
                "a/b/c/cache.tmp": "",
                "src/build/out.o": "",
                "build/out.o": "",
                "foo/a/b/bar": "",
                "foo/a/bar": "",
                "foo/bar": "",
                "one/a/b/bar": "",
                "one/a/bar": "",
                ".dockerignore": "**/*.tmp\n/build\nfoo/**/bar\none/*/bar\n",
            },
            [
                ".dockerignore",
                "Dockerfile.hud",
                "a",
                "a/b",
                "a/b/c",
                "env.py",
                "foo",
                "foo/a",
                "foo/a/b",
                "one",
                "one/a",
                "one/a/b",
                "one/a/b/bar",
                "src",
                "src/build",
                "src/build/out.o",
            ],
            id="double-star-anchored-and-single-star",
        ),
    ],
)
def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    hud: Hud, builds: FakeServices, files: dict[str, str], expected: list[str]
) -> None:
    for name, text in files.items():
        path = hud.cwd / name
        if name.endswith("/"):
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert uploaded(builds) == expected


def test_deploying_a_file_uploads_its_directory(hud: Hud, builds: FakeServices) -> None:
    environment(hud.cwd, "adapter")
    environment(hud.cwd / "source", "original")
    (hud.cwd / "compose.yaml").write_text("services: {}\n")

    result = hud("deploy", "env.py", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert trigger_body(builds)["name"] == "adapter"
    assert {"env.py", "compose.yaml", "source/env.py"} <= set(uploaded(builds))


@pytest.mark.parametrize(
    ("files", "argv", "name"),
    [
        ({"env.py": 'env = Environment("my-env")\n'}, [], "my-env"),
        (
            {
                "a.py": 'a = Environment("same")\n',
                "b.py": 'b = Environment(name="same")\n',
                "c.py": 'c = hud.Environment("same")\n',
            },
            [],
            "same",
        ),
        (
            {
                "server/pkg/env.py": 'env = Environment("nested")\n',
                ".venv/env.py": 'env = Environment("excluded")\n',
                "node_modules/env.py": 'env = Environment("excluded")\n',
                "__pycache__/env.py": 'env = Environment("excluded")\n',
                "broken.py": "def broken(:\n",
            },
            [],
            "nested",
        ),
        (
            {"env.py": 'env = Environment("named")\nother = Environment(name=NAME)\n'},
            [],
            "named",
        ),
        (
            {"env.py": 'actor = Environment("workspace")\nverifier = Environment("judge")\n'},
            ["--name", "judge"],
            "judge",
        ),
        (
            {
                "Dockerfile": 'CMD ["hud", "serve", "env:env"]\n',
                "env.py": 'env = Environment("trace-explorer")\n',
                "verify.py": 'verify_env = Environment("qa-verifier")\n',
            },
            ["--name", "trace-explorer"],
            "trace-explorer",
        ),
        (
            {"env.py": 'env = Environment("adapter")\n', "source/env.py": 'Environment("x")\n'},
            ["env.py"],
            "adapter",
        ),
    ],
)
def test_the_environment_name_comes_from_its_declaration(
    hud: Hud, projects: FakeServices, files: dict[str, str], argv: list[str], name: str
) -> None:
    for path, text in files.items():
        (hud.cwd / path).parent.mkdir(parents=True, exist_ok=True)
        (hud.cwd / path).write_text(text)

    result = hud("deploy", *argv, "--dry-run", "--no-env", "--json")

    assert result.exit_code == 0, result
    assert result.json["name"] == name


@pytest.mark.parametrize(
    ("setup", "argv", "plan"),
    [
        (
            {},
            [],
            snapshot(
                {
                    "success": True,
                    "action": "deploy",
                    "build_id": None,
                    "registry_id": None,
                    "status": "",
                    "name": "example",
                    "dry_run": True,
                    "runtime": None,
                    "env_var_keys": [],
                    "build_arg_keys": [],
                    "dotenv_pending": False,
                    "details": {},
                }
            ),
        ),
        (
            {},
            ["--runtime", "MODAL", "--build-arg", "A=1", "--env", "K=V"],
            snapshot(
                {
                    "success": True,
                    "action": "deploy",
                    "build_id": None,
                    "registry_id": None,
                    "status": "",
                    "name": "example",
                    "dry_run": True,
                    "runtime": "modal",
                    "env_var_keys": ["K"],
                    "build_arg_keys": ["A"],
                    "dotenv_pending": False,
                    "details": {},
                }
            ),
        ),
        (
            {"vars.env": "KEY1=v\nEMPTY=\nNOEQ\n"},
            ["--env-file", "vars.env"],
            snapshot(
                {
                    "success": True,
                    "action": "deploy",
                    "build_id": None,
                    "registry_id": None,
                    "status": "",
                    "name": "example",
                    "dry_run": True,
                    "runtime": None,
                    "env_var_keys": ["EMPTY", "KEY1"],
                    "build_arg_keys": [],
                    "dotenv_pending": False,
                    "details": {},
                }
            ),
        ),
        (
            {".env": "SECRET=hidden", ".hud/deploy.json": '{"registryId":"old"}'},
            [],
            snapshot(
                {
                    "success": True,
                    "action": "deploy",
                    "build_id": None,
                    "registry_id": None,
                    "status": "",
                    "name": "example",
                    "dry_run": True,
                    "runtime": None,
                    "env_var_keys": [],
                    "build_arg_keys": [],
                    "dotenv_pending": True,
                    "details": {},
                }
            ),
        ),
    ],
)
def test_a_dry_run_plans_without_prompting_or_writing(
    hud: Hud,
    projects: FakeServices,
    hud_env: HudEnv,
    setup: dict[str, str],
    argv: list[str],
    plan: dict[str, Any],
) -> None:
    environment(hud.cwd)
    for name, text in setup.items():
        (hud.cwd / name).parent.mkdir(parents=True, exist_ok=True)
        (hud.cwd / name).write_text(text)
    before = sorted(path.name for path in hud.cwd.rglob("*"))

    result = hud("deploy", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json == plan
    assert sorted(path.name for path in hud.cwd.rglob("*")) == before
    assert not (hud_env.home / ".hud").exists()
    assert projects.requests("api", "POST") == []


@pytest.mark.parametrize(
    ("runtime_config", "expected"),
    [
        (
            {"resources": {"gpu": {"type": "A10G", "count": 2}}},
            snapshot({"resources": {"gpu": {"type": "A10G", "count": 2}}}),
        ),
        ({"resources": None}, snapshot({"resources": None})),
        (
            {"compose": {"document": "project/compose.json", "root": "."}},
            snapshot(
                {
                    "compose": {
                        "document": {
                            "services": {
                                "main": {
                                    "image": "postgres:16",
                                    "environment": {},
                                    "expose": [],
                                    "ports": [],
                                    "volumes": [],
                                }
                            },
                            "networks": {},
                        },
                        "root": {"compose_path": "project/compose.json"},
                    }
                }
            ),
        ),
    ],
)
def test_the_runtime_config_reaches_the_trigger_in_sdk_shape(
    hud: Hud, builds: FakeServices, runtime_config: dict[str, Any], expected: dict[str, Any]
) -> None:
    environment(hud.cwd)
    (hud.cwd / "project").mkdir()
    (hud.cwd / "project" / "compose.json").write_text(
        '{"services":{"main":{"image":"postgres:16"}}}'
    )
    (hud.cwd / "runtime.json").write_text(json.dumps(runtime_config))

    result = hud("deploy", "--no-env", "--runtime-config", "runtime.json", "--json")

    assert result.exit_code == 0, result
    assert trigger_body(builds)["runtime_config"] == expected


def test_a_plain_deploy_sends_only_the_build(hud: Hud, builds: FakeServices) -> None:
    environment(hud.cwd)
    (hud.cwd / "compose.yaml").write_text("services:\n  main:\n    image: alpine\n")

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
)
def test_a_redeploy_keeps_the_link_and_the_registry_secrets(
    hud: Hud,
    builds: FakeServices,
    second: list[str],
    body: dict[str, Any],
    kept: bool,
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
    first_body, second_body = builds.bodies("api", "POST", "/v2/builds/trigger")
    assert first_body["environment_variables"] == {"SECRET": "dotenv"}
    assert {key: second_body[key] for key in second_body if key not in FIXED} == body
    assert ("Registry secrets: kept" in again.stderr) is kept


# ─── failures ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("files", "argv", "exit_code", "document"),
    [
        pytest.param(
            {},
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "No environment found in <path0>.",
                    "suggestion": (
                        "Declare the environment with Environment(name=...) in a .py file."
                    ),
                }
            ),
            id="empty-tree",
        ),
        pytest.param(
            {"env.py": "env = Environment(name=NAME)\n"},
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "No environment found in <path0>.",
                    "suggestion": (
                        "Declare the environment with Environment(name=...) in a .py file."
                    ),
                }
            ),
            id="no-literal",
        ),
        pytest.param(
            {"selected.py": "x = 1\n", "env.py": 'env = Environment("other")\n'},
            ["selected.py"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "No environment found in <path0>/selected.py.",
                    "suggestion": (
                        "Declare the environment with Environment(name=...) in a .py file."
                    ),
                }
            ),
            id="file-without-declaration",
        ),
        pytest.param(
            {"server.py": "x = 1\n"},
            ["--registry-id", REGISTRY_ID],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "No environment found in <path0>.",
                    "suggestion": (
                        "Declare the environment with Environment(name=...) in a .py file."
                    ),
                }
            ),
            id="registry-id-without-declaration",
        ),
        pytest.param(
            {"a.py": 'a = Environment("a")\n', "b.py": 'b = Environment("b")\n'},
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Multiple environments in <path0>: a, b. Pass --name to choose one.",
                }
            ),
            id="ambiguous-name",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--name", "missing"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "No environment named 'missing' in <path0>. Found: a.",
                }
            ),
            id="unknown-name",
        ),
        pytest.param(
            {"env.py": 'env = Environment("My Env")\n'},
            ["--registry-id", REGISTRY_ID],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Code declares Environment('My Env') but --registry-id targets 'other-env'."
                    ),
                    "input": {
                        "registry_id": "<uuid>",
                        "name": "My Env",
                    },
                    "suggestion": (
                        "Rename the environment in code, or drop --registry-id to deploy by name."
                    ),
                }
            ),
            id="registry-id-names-another-environment",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--registry-id", OTHER_REGISTRY_ID],
            1,
            snapshot(
                {
                    "error": "not_found",
                    "message": "Request failed: gone",
                    "input": {"registry_id": "<uuid>"},
                    "suggestion": "Check the environment id, or list existing ones.",
                }
            ),
            id="unknown-registry-id",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--runtime", "moddal"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Unknown runtime 'moddal'. Choose one of: hud, modal.",
                    "input": {"runtime": "moddal"},
                }
            ),
            id="runtime",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--env", "INVALID"],
            2,
            snapshot(
                {"error": "usage", "message": "Invalid --env format: INVALID (expected KEY=VALUE)"}
            ),
            id="env",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--build-arg", "NOPE"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Invalid --build-arg format: NOPE (expected KEY=VALUE)",
                }
            ),
            id="build-arg",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--secret", "env=X"],
            2,
            snapshot({"error": "usage", "message": "Invalid --secret format: env=X (missing id=)"}),
            id="secret-id",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--secret", "id=X,env=HUD_UNSET_SECRET"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": "Secret 'X': environment variable 'HUD_UNSET_SECRET' is not set",
                }
            ),
            id="secret-env-unset",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--secret", "id=X,src=missing.txt"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Secret 'X': failed to read <path0>/missing.txt: [Errno 2] No such file or "
                        "directory: '<path0>/missing.txt'"
                    ),
                }
            ),
            id="secret-src-missing",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--secret", "id=X"],
            2,
            snapshot(
                {"error": "usage", "message": "Invalid --secret format: id=X (need env= or src=)"}
            ),
            id="secret-kind",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--env-file", "missing.env"],
            1,
            snapshot({"error": "not_found", "message": "Env file not found: missing.env"}),
            id="missing-env-file",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n', "runtime.json": "{}"},
            ["--runtime-config", "runtime.json"],
            2,
            snapshot(
                {"error": "usage", "message": "--runtime-config must set at least one field."}
            ),
            id="empty-runtime-config",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n', "runtime.json": '{"provider_config": {}}'},
            ["--runtime-config", "runtime.json"],
            2,
            {
                "error": "usage",
                # pydantic's help link names its own version
                "message": IsStr(
                    regex=r"1 validation error for RuntimeConfig\nprovider_config\n"
                    r"  Extra inputs are not permitted \[type=extra_forbidden, .*",
                    regex_flags=re.DOTALL,
                ),
            },
            id="unknown-runtime-config-field",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n', ".env": "SECRET=1"},
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Choose whether to seed registry secrets from .env before deploying."
                    ),
                    "suggestion": "Pass --env-file .env to upsert them, or --no-env to skip.",
                }
            ),
            id="first-deploy-in-ci-with-dotenv",
        ),
        pytest.param(
            {"env.py": 'env = Environment("a")\n'},
            ["--project", LOCKED_PROJECT_ID],
            1,
            snapshot(
                {
                    "error": "permission_denied",
                    "message": (
                        "You do not have permission to create environments or tasksets in project "
                        "'locked-down'"
                    ),
                    "input": {"project": "<uuid>"},
                }
            ),
            id="read-only-project",
        ),
    ],
)
def test_deploy_refuses_before_uploading(
    hud: Hud,
    projects: FakeServices,
    files: dict[str, str],
    argv: list[str],
    exit_code: int,
    document: dict[str, Any],
) -> None:
    projects.route(
        "api", "GET", f"/v2/registry/{REGISTRY_ID}", json={"id": REGISTRY_ID, "name": "other-env"}
    )
    projects.route(
        "api", "GET", f"/v2/registry/{OTHER_REGISTRY_ID}", status=404, json={"detail": "gone"}
    )
    for name, text in files.items():
        (hud.cwd / name).write_text(text)

    result = hud("deploy", *argv, "--json")

    assert result.exit_code == exit_code, result
    assert json.loads(scrub(result.stdout, hud.cwd)) == document
    assert projects.requests("api", "POST") == []


def test_deploy_without_a_key_is_permission_denied(hud: Hud) -> None:
    environment(hud.cwd)

    result = hud("deploy", "--json")

    assert result.exit_code == 1
    assert result.json["error"] == "permission_denied"


@pytest.mark.parametrize(
    ("path", "reply", "document"),
    [
        (
            "/v2/builds/upload-url",
            Reply(status=401, json={"detail": "Unauthorized"}),
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: Unauthorized",
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
        ),
        (
            "/v2/builds/trigger",
            Reply(status=409, json={"detail": "Build running"}),
            snapshot({"error": "conflict", "message": "Request failed: Build running"}),
        ),
        (
            "/v2/builds/trigger",
            Reply(status=429, json={"detail": "Slow down"}),
            snapshot(
                {
                    "error": "rate_limited",
                    "message": "Request failed: Slow down",
                    "suggestion": "Retry after a short delay.",
                }
            ),
        ),
        (
            "/v2/builds/trigger",
            Reply(status=500, json={"detail": "Boom"}),
            snapshot(
                {
                    "error": "server_error",
                    "message": "Request failed: Boom",
                    "suggestion": "Retry; this error is often transient.",
                }
            ),
        ),
    ],
)
def test_a_rejected_build_request_is_an_error_document(
    hud: Hud,
    builds: FakeServices,
    tmp_path: Path,
    path: str,
    reply: Reply,
    document: dict[str, Any],
) -> None:
    builds.route("api", "POST", path, reply)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == 1, result
    assert result.json == document
    assert list((tmp_path / "tmp").iterdir()) == []
    assert not (hud.cwd / ".hud").exists()


@pytest.mark.parametrize(
    ("status", "stream", "exit_code", "stderr"),
    [
        pytest.param(
            {"status": "FAILED", "error_message": "pip install failed"},
            log_stream([{"type": "error", "error": "Step 2 failed"}]),
            1,
            ["Build error: Step 2 failed", "pip install failed"],
            id="failed-build",
        ),
        pytest.param(
            SUCCEEDED,
            log_stream([], close=(4003, "token expired")),
            0,
            ["Access denied: token expired"],
            id="access-denied",
        ),
        pytest.param(
            SUCCEEDED,
            log_stream([], close=(1011, "restarting")),
            0,
            ["Log stream closed: restarting"],
            id="stream-closed",
        ),
        pytest.param(SUCCEEDED, refuse, 0, ["Log stream unavailable"], id="no-stream"),
    ],
)
def test_the_build_status_decides_the_exit_whatever_the_log_stream_does(
    hud: Hud,
    builds: FakeServices,
    status: dict[str, Any],
    stream: Callable[[WebSocket, dict[str, str]], Awaitable[None]],
    exit_code: int,
    stderr: list[str],
) -> None:
    builds.route("api", "GET", "/v2/builds/build-1/status", json=status)
    builds.websocket("api", "/v2/builds/build-1/logs", stream)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--json")

    assert result.exit_code == exit_code, result
    assert result.json["status"] == status["status"]
    assert result.json["success"] is (exit_code == 0)
    assert all(line in result.stderr for line in stderr), result.stderr
