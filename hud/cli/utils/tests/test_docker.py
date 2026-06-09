"""Pure helpers in ``hud.cli.utils.docker`` (no Docker daemon needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from hud.cli.utils import docker

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_extract_name_and_tag() -> None:
    assert docker.extract_name_and_tag("hudpython/myenv:v1.0") == ("hudpython/myenv", "v1.0")
    assert docker.extract_name_and_tag("myorg/myapp") == ("myorg/myapp", "latest")
    assert docker.extract_name_and_tag("docker.io/org/img:tag@sha256:abc") == ("org/img", "tag")


def test_generate_container_name_sanitizes() -> None:
    assert docker.generate_container_name("org/img:tag") == "hud-org-img-tag"
    assert docker.generate_container_name("x", prefix="run") == "run-x"


def test_build_run_command() -> None:
    assert docker.build_run_command("img") == ["docker", "run", "--rm", "-i", "img"]
    assert docker.build_run_command("img", ["-e", "K=V"]) == [
        "docker",
        "run",
        "--rm",
        "-i",
        "-e",
        "K=V",
        "img",
    ]


def test_build_env_flags() -> None:
    assert docker.build_env_flags({"A": "1", "B": "2"}) == ["-e", "A=1", "-e", "B=2"]


def test_normalize_cmd_handles_exec_and_shell_forms() -> None:
    assert docker._normalize_cmd(["hud", "dev", "env:env"]) == ["hud", "dev", "env:env"]
    assert docker._normalize_cmd(["sh", "-c", "hud dev env:env --port 8080"]) == [
        "hud",
        "dev",
        "env:env",
        "--port",
        "8080",
    ]


def test_detect_transport_http_with_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        docker, "get_docker_cmd", lambda _img: ["hud", "dev", "env:env", "--port", "9000"]
    )
    assert docker.detect_transport("img") == ("http", 9000)


def test_detect_transport_defaults_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker, "get_docker_cmd", lambda _img: ["python", "server.py"])
    assert docker.detect_transport("img") == ("stdio", None)


def test_detect_transport_no_cmd_is_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker, "get_docker_cmd", lambda _img: None)
    assert docker.detect_transport("img") == ("stdio", None)


def test_detect_environment_dir_finds_lockfile(tmp_path: Path) -> None:
    (tmp_path / "hud.lock.yaml").write_text("version: '2.0'\n", encoding="utf-8")
    assert docker.detect_environment_dir(tmp_path) == tmp_path


def test_detect_environment_dir_falls_back_to_dockerfile(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    assert docker.detect_environment_dir(tmp_path) == tmp_path


def test_load_env_vars_for_dir(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("KEY=value\nOTHER=2\n", encoding="utf-8")
    assert docker.load_env_vars_for_dir(tmp_path) == {"KEY": "value", "OTHER": "2"}


def test_load_env_vars_missing_is_empty(tmp_path: Path) -> None:
    assert docker.load_env_vars_for_dir(tmp_path) == {}


def test_image_exists_true() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert docker.image_exists("img") is True
