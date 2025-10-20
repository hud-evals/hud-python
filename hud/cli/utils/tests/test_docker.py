from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.cli.utils.docker import (
    build_run_command,
    generate_container_name,
    get_docker_cmd,
    image_exists,
    remove_container,
    require_docker_running,
)


def test_build_run_command_basic():
    cmd = build_run_command("my-image:latest")
    assert cmd[:4] == ["docker", "run", "--rm", "-i"]
    assert cmd[-1] == "my-image:latest"


def test_build_run_command_with_args():
    cmd = build_run_command("img", ["-e", "K=V", "-p", "8080:8080"])
    assert "-e" in cmd and "K=V" in cmd
    assert "-p" in cmd and "8080:8080" in cmd
    assert cmd[-1] == "img"


def test_generate_container_name():
    assert generate_container_name("repo/name:tag") == "hud-repo-name-tag"
    assert generate_container_name("a/b:c", prefix="x") == "x-a-b-c"


@patch("subprocess.run")
def test_image_exists_true(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert image_exists("any") is True


@patch("subprocess.run")
def test_image_exists_false(mock_run):
    mock_run.return_value = MagicMock(returncode=1)
    assert image_exists("any") is False


@patch("subprocess.run")
def test_get_docker_cmd_success(mock_run):
    mock_run.return_value = MagicMock(
        stdout='[{"Config": {"Cmd": ["python", "-m", "app"]}}]', returncode=0
    )
    assert get_docker_cmd("img") == ["python", "-m", "app"]


@patch("subprocess.run")
def test_get_docker_cmd_none(mock_run):
    mock_run.return_value = MagicMock(stdout="[]", returncode=0)
    assert get_docker_cmd("img") is None


@patch("subprocess.run")
def test_remove_container_ok(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert remove_container("x") is True


@patch("shutil.which", return_value=None)
def test_require_docker_running_no_cli(_which):
    import typer

    with pytest.raises(typer.Exit):
        require_docker_running()


@patch("shutil.which", return_value="docker")
@patch("subprocess.run")
def test_require_docker_running_ok(mock_run, _which):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    require_docker_running()  # should not raise


@patch("shutil.which", return_value="docker")
@patch("subprocess.run")
def test_require_docker_running_error_emits_hints(mock_run, _which):
    import typer

    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="Cannot connect to the Docker daemon",
        stderr="",
    )
    with pytest.raises(typer.Exit):
        require_docker_running()
