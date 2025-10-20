from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

from hud.cli.build import (
    extract_env_vars_from_dockerfile,
    get_docker_image_digest,
    get_docker_image_id,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_env_vars_from_dockerfile_complex(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        """
FROM python:3.11
ARG BUILD_TOKEN
ARG DEFAULTED=1
ENV RUNTIME_KEY
ENV FROM_ARG=$BUILD_TOKEN
ENV WITH_DEFAULT=val
"""
    )
    required, optional = extract_env_vars_from_dockerfile(dockerfile)
    # BUILD_TOKEN required (ARG without default)
    assert "BUILD_TOKEN" in required
    # RUNTIME_KEY required (ENV without value)
    assert "RUNTIME_KEY" in required
    # FROM_ARG references BUILD_TOKEN -> required
    assert "FROM_ARG" in required
    # DEFAULTED and WITH_DEFAULT should not be marked required by default
    assert "DEFAULTED" not in required
    assert "WITH_DEFAULT" not in required
    assert optional == []


@mock.patch("subprocess.run")
def test_get_docker_image_digest_none(mock_run):
    mock_run.return_value = mock.Mock(stdout="[]", returncode=0)
    assert get_docker_image_digest("img") is None


@mock.patch("subprocess.run")
def test_get_docker_image_id_ok(mock_run):
    mock_run.return_value = mock.Mock(stdout="sha256:abc", returncode=0)
    assert get_docker_image_id("img") == "sha256:abc"
