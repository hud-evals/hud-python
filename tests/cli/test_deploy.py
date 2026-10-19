"""``hud deploy``: the build context it uploads, the build it triggers, and how it reports it."""

from __future__ import annotations

import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply, write_files
from tests.support.platform import BROWSER_PROJECT_ID, Build

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.support import Hud, HudEnv, Platform

ENV_PY = 'from hud import Environment\nenv = Environment("example")\n'


@pytest.fixture
def tmpdir_(tmp_path: Path, hud_env: HudEnv) -> Path:
    """A private ``TMPDIR``, so a test can see that the build context tarball is removed."""
    (tmp_path / "tmp").mkdir()
    hud_env.set(TMPDIR=str(tmp_path / "tmp"))
    return tmp_path / "tmp"


def environment(directory: Path, **files: str) -> Path:
    write_files(directory, {"env.py": ENV_PY, "Dockerfile.hud": "FROM python:3.12\n", **files})
    return directory


def context(build: Build) -> list[str]:
    """Member names of the build context tarball the platform received."""
    assert build.context is not None
    with tarfile.open(fileobj=io.BytesIO(build.context), mode="r:gz") as archive:
        return sorted(archive.getnames())


def trigger(platform: Platform) -> dict[str, Any]:
    """The trigger of the last build the platform was asked for."""
    *_, build = platform.builds.values()
    assert build.trigger is not None
    return build.trigger


def settings(hud: Hud, platform: Platform) -> Any:
    found = trigger(platform)
    return (found["environment_variables"], found["build_args"], found["build_secrets"])


def compose(hud: Hud, platform: Platform) -> Any:
    document = trigger(platform)["runtime_config"]["compose"]
    return (document["document"]["services"]["main"]["image"], document["root"])


def placed(hud: Hud, platform: Platform) -> Any:
    (example,) = platform.environments.values()
    linked = json.loads((hud.cwd / ".hud" / "config.json").read_text())["registry_id"]
    return (example["project_id"], linked == example["id"])


def redeployed(hud: Hud, platform: Platform) -> Any:
    first, second = platform.builds.values()
    assert first.trigger is not None and second.trigger is not None
    return ("environment_variables" in first.trigger, "environment_variables" in second.trigger)


SENT: dict[str, tuple[dict[str, str], list[list[str]], dict[str, str], Callable[..., Any], Any]] = {
    "env files, build args and secrets ride the trigger": (
        {".env": "SECRET=dotenv-secret\n", "token.txt": "file-secret", "vars.env": "KEY=v\n"},
        [
            [
                *("--env-file", "vars.env", "--build-arg", "PYTHON=3.12"),
                *("--secret", "id=GITHUB_TOKEN,env=GITHUB_TOKEN"),
                *("--secret", "id=FILE_TOKEN,src=token.txt"),
            ]
        ],
        {"GITHUB_TOKEN": "env-secret"},
        settings,
        (
            {"KEY": "v"},
            {"PYTHON": "3.12"},
            {"GITHUB_TOKEN": "env-secret", "FILE_TOKEN": "file-secret"},
        ),
    ),
    "a Compose runtime config reaches the build as its document": (
        {
            "project/compose.json": '{"services":{"main":{"image":"postgres:16"}}}',
            "runtime.json": json.dumps(
                {"compose": {"document": "project/compose.json", "root": "."}}
            ),
        },
        [["--no-env", "--runtime-config", "runtime.json"]],
        {},
        compose,
        ("postgres:16", {"compose_path": "project/compose.json"}),
    ),
    "a first deploy lands in the selected project and links the directory": (
        {},
        [["--no-env"]],
        {"HUD_DEFAULT_PROJECT": "browser-evals"},
        placed,
        (BROWSER_PROJECT_ID, True),
    ),
    "a redeploy keeps the registry's variables": (
        {".env": "SECRET=dotenv"},
        [["--env-file", ".env"], []],
        {},
        redeployed,
        (True, False),
    ),
}


@pytest.mark.parametrize(
    ("files", "deploys", "env", "observe", "expected"), SENT.values(), ids=SENT.keys()
)
def test_a_deploy_sends_what_its_options_name_with_the_build(
    hud: Hud,
    platform: Platform,
    files: dict[str, str],
    deploys: list[list[str]],
    env: dict[str, str],
    observe: Callable[[Hud, Platform], Any],
    expected: Any,
) -> None:
    environment(hud.cwd, **files)

    results = [hud("deploy", *argv, "--yes", "--json", env=env) for argv in deploys]

    assert [result.exit_code for result in results] == [0] * len(deploys), results[-1]
    assert observe(hud, platform) == expected
    for build in platform.builds.values():
        assert ".env" not in context(build)


def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    hud: Hud, platform: Platform
) -> None:
    environment(
        hud.cwd,
        **{
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
    (build,) = platform.builds.values()
    sent = set(context(build))
    assert {"keep.pyc", "src/build/out.o"} <= sent
    assert not {"drop.pyc", "build/out.o", ".git/config", "nested/.env.prod"} & sent


@pytest.mark.parametrize(
    ("files", "argv", "planned"),
    [
        pytest.param(
            {"env.py": 'env = Environment("my-env")\n'}, [], {"name": "my-env"}, id="declared"
        ),
        pytest.param(
            {"env.py": 'actor = Environment("workspace")\nverifier = Environment("judge")\n'},
            ["--name", "judge"],
            {"name": "judge"},
            id="chosen-among-several",
        ),
        pytest.param(
            {"env.py": ENV_PY, ".env": "SECRET=hidden"},
            [],
            {"name": "example", "dotenv_pending": True},
            id="a-dotenv-beside-it-is-asked-about",
        ),
    ],
)
def test_a_dry_run_plans_the_environment_it_would_deploy(
    hud: Hud, platform: Platform, files: dict[str, str], argv: list[str], planned: dict[str, Any]
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", *argv, "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert {key: result.json[key] for key in planned} == planned


ENDINGS: dict[str, tuple[Build | None, Reply | None, int, str]] = {
    "a build that succeeds streams its log": (None, None, 0, "Step 1/2 : FROM python:3.12"),
    "a build that fails": (
        Build(
            "",
            status={"status": "FAILED", "error_message": "pip install failed"},
            log=[{"type": "error", "error": "Step 2 failed"}],
        ),
        None,
        1,
        "",
    ),
    "a build without a log stream": (
        Build("", status={"status": "SUCCEEDED"}, log=None),
        None,
        0,
        "",
    ),
    "a build request the platform rejects": (
        None,
        Reply(status=409, json={"detail": "Busy"}),
        1,
        "",
    ),
}


@pytest.mark.parametrize(
    ("build", "rejected", "exit_code", "logged"), ENDINGS.values(), ids=ENDINGS.keys()
)
def test_the_build_decides_the_exit_and_no_context_is_left_behind(
    hud: Hud,
    platform: Platform,
    tmpdir_: Path,
    build: Build | None,
    rejected: Reply | None,
    exit_code: int,
    logged: str,
) -> None:
    if build is not None:
        platform.next_build = build
    if rejected is not None:
        platform.services.route("api", "POST", "/v2/builds/trigger", rejected)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert logged in result.stderr
    if rejected is None:
        assert result.json["success"] is (exit_code == 0)
    assert list(tmpdir_.iterdir()) == []
