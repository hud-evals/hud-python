from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from hud.cli.push import push_environment

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.push.get_docker_username", return_value="tester")
@patch(
    "hud.cli.push.get_docker_image_labels",
    return_value={"org.hud.manifest.head": "abc", "org.hud.version": "0.1.0"},
)
@patch("hud.cli.push.requests.post")
@patch("hud.cli.push.subprocess.Popen")
@patch("hud.cli.push.subprocess.run")
def test_push_happy_path(
    mock_run, mock_popen, mock_post, _labels, _user, tmp_path: Path, monkeypatch
):
    # Prepare minimal environment with lock file
    env_dir = tmp_path
    (env_dir / "hud.lock.yaml").write_text(
        "images:\n  local: org/env:latest\nbuild:\n  version: 0.1.0\n"
    )

    # Provide API key via main settings module
    monkeypatch.setattr("hud.settings.settings.api_key", "sk-test", raising=False)

    # ensure_built noop - patch from the right module
    monkeypatch.setattr("hud.cli.utils.env_check.ensure_built", lambda *_a, **_k: {})

    # Mock subprocess.run behavior depending on command
    def run_side_effect(args, *a, **k):
        cmd = list(args)
        # docker inspect checks
        if cmd[:2] == ["docker", "inspect"]:
            # For label digest query at end
            if "--format" in cmd and "{{index .RepoDigests 0}}" in cmd:
                return SimpleNamespace(returncode=0, stdout="org/env@sha256:deadbeef")
            # Existence checks succeed
            return SimpleNamespace(returncode=0, stdout="")
        # docker tag success
        if cmd[:2] == ["docker", "tag"]:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout="")

    mock_run.side_effect = run_side_effect

    # Mock Popen push pipeline
    class _Proc:
        def __init__(self):
            self.stdout = ["digest: sha256:deadbeef\n", "pushed\n"]
            self.returncode = 0

        def wait(self):
            return 0

    mock_popen.return_value = _Proc()

    # Mock registry POST success
    mock_post.return_value = SimpleNamespace(status_code=201, json=lambda: {"ok": True}, text="")

    # Execute
    push_environment(
        directory=str(env_dir), image=None, tag=None, sign=False, yes=True, verbose=False
    )

    # Lock file updated with pushed entry
    data = (env_dir / "hud.lock.yaml").read_text()
    assert "pushed:" in data
