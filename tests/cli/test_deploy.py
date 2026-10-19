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
    from pathlib import Path

    from tests.support import Hud, HudEnv, Platform


@pytest.fixture
def tmpdir_(tmp_path: Path, hud_env: HudEnv) -> Path:
    """A private ``TMPDIR``, so a test can see that the build context tarball is removed."""
    (tmp_path / "tmp").mkdir()
    hud_env.set(TMPDIR=str(tmp_path / "tmp"))
    return tmp_path / "tmp"


def environment(directory: Path, name: str = "example") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "env.py").write_text(f'from hud import Environment\nenv = Environment("{name}")\n')
    (directory / "Dockerfile.hud").write_text("FROM python:3.12\n")
    return directory


def built(platform: Platform) -> Build:
    (build,) = platform.builds.values()
    return build


def context(build: Build) -> list[str]:
    """Member names of the build context tarball the platform received."""
    assert build.context is not None
    with tarfile.open(fileobj=io.BytesIO(build.context), mode="r:gz") as archive:
        return sorted(archive.getnames())


def test_deploy_uploads_the_context_triggers_the_build_and_links_the_directory(
    hud: Hud, platform: Platform, tmpdir_: Path
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
        *("--secret", "id=GITHUB_TOKEN,env=GITHUB_TOKEN"),
        *("--secret", "id=FILE_TOKEN,src=token.txt"),
        "--yes",
        env={"GITHUB_TOKEN": "env-secret"},
    )

    assert result.exit_code == 0, result
    assert "Step 1/2 : FROM python:3.12" in result.stderr  # the build log streams
    build = built(platform)
    assert build.trigger is not None
    assert (build.trigger["environment_variables"], build.trigger["build_args"]) == (
        {"KEY": "file_value"},
        {"PYTHON": "3.12"},
    )
    assert build.trigger["build_secrets"] == {
        "GITHUB_TOKEN": "env-secret",
        "FILE_TOKEN": "file-secret",
    }
    assert ".env" not in context(build)
    assert list(tmpdir_.iterdir()) == []
    (example,) = platform.environments.values()
    assert json.loads((env / ".hud" / "config.json").read_text())["registry_id"] == example["id"]


def test_the_build_context_follows_dockerignore_and_never_holds_secrets(
    hud: Hud, platform: Platform
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

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == 0, result
    sent = set(context(built(platform)))
    assert {"keep.pyc", "src/build/out.o"} <= sent
    assert not {"drop.pyc", "build/out.o", ".git/config", "nested/.env.prod"} & sent


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
    hud: Hud, platform: Platform, files: dict[str, str], argv: list[str], name: str
) -> None:
    write_files(hud.cwd, files)

    result = hud("deploy", *argv, "--dry-run", "--no-env", "--json")

    assert (result.exit_code, result.json["name"]) == (0, name), result


def test_a_dry_run_plans_to_ask_about_a_dotenv_beside_the_environment(
    hud: Hud, platform: Platform
) -> None:
    environment(hud.cwd)
    (hud.cwd / ".env").write_text("SECRET=hidden")

    result = hud("deploy", "--dry-run", "--json")

    assert (result.exit_code, result.json["dotenv_pending"]) == (0, True), result


def test_a_compose_runtime_config_reaches_the_build_as_its_document(
    hud: Hud, platform: Platform
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

    result = hud("deploy", "--no-env", "--runtime-config", "runtime.json", "--yes", "--json")

    assert result.exit_code == 0, result
    trigger = built(platform).trigger
    assert trigger is not None
    compose = trigger["runtime_config"]["compose"]
    assert compose["document"]["services"]["main"]["image"] == "postgres:16"
    assert compose["root"] == {"compose_path": "project/compose.json"}


def test_a_redeploy_keeps_the_link_and_the_registry_secrets(hud: Hud, platform: Platform) -> None:
    environment(hud.cwd)
    (hud.cwd / ".env").write_text("SECRET=dotenv")

    first = hud("deploy", "--env-file", ".env", "--yes", "--json")
    linked = (hud.cwd / ".hud" / "config.json").read_bytes()
    again = hud("deploy", "--yes", "--json")

    assert (first.exit_code, again.exit_code) == (0, 0), again
    assert (hud.cwd / ".hud" / "config.json").read_bytes() == linked
    _, redeployed = platform.builds.values()
    assert redeployed.trigger is not None
    assert "environment_variables" not in redeployed.trigger


def test_a_first_deploy_lands_in_the_selected_project(
    hud: Hud, platform: Platform, hud_env: HudEnv
) -> None:
    environment(hud.cwd)
    hud_env.set(HUD_DEFAULT_PROJECT="browser-evals")

    result = hud("deploy", "--yes", "--json")

    assert result.exit_code == 0, result
    (example,) = platform.environments.values()
    assert example["project_id"] == BROWSER_PROJECT_ID


def test_a_rejected_build_request_removes_its_build_context(
    hud: Hud, platform: Platform, tmpdir_: Path
) -> None:
    platform.services.route(
        "api", "POST", "/v2/builds/trigger", Reply(status=409, json={"detail": "Busy"})
    )
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == 1, result
    assert list(tmpdir_.iterdir()) == []


@pytest.mark.parametrize(
    ("status", "log", "exit_code"),
    [
        pytest.param(
            {"status": "FAILED", "error_message": "pip install failed"},
            [{"type": "error", "error": "Step 2 failed"}],
            1,
            id="failed-build",
        ),
        pytest.param({"status": "SUCCEEDED"}, None, 0, id="no-log-stream"),
    ],
)
def test_the_build_status_decides_the_exit_whatever_the_log_stream_does(
    hud: Hud,
    platform: Platform,
    status: dict[str, Any],
    log: list[dict[str, Any]] | None,
    exit_code: int,
) -> None:
    platform.next_build = Build("", status=status, log=log)
    environment(hud.cwd)

    result = hud("deploy", "--no-env", "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert result.json["success"] is (exit_code == 0)
