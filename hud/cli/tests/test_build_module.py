from __future__ import annotations

from unittest import mock

from hud.cli.build import get_docker_image_id


@mock.patch("subprocess.run")
def test_get_docker_image_id_ok(mock_run):
    mock_run.return_value = mock.Mock(stdout="sha256:abc", returncode=0)
    assert get_docker_image_id("img") == "sha256:abc"
