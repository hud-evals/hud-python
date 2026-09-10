"""Make the environment project importable when tests run from the SDK root."""

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from hud.environment import Workspace
from hud.environment.workspace import usable_bwrap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_workspace():
    if sys.platform != "linux" or os.geteuid() != 0 or usable_bwrap() is None:
        pytest.skip("requires Linux, root, and bubblewrap (run in the coding container)")


@pytest_asyncio.fixture(loop_scope="session")
async def grading_workspace(tmp_path, monkeypatch, isolated_workspace):
    import env as coding_env

    root = tmp_path / "repo"
    workspace = Workspace(root, guest_path=str(root), network=False, shell_uid=1000, require_isolation=True)
    await workspace.start()
    monkeypatch.setattr(coding_env, "workspace", workspace)
    monkeypatch.setattr(coding_env, "REPO_DIR", root)
    monkeypatch.setattr(coding_env, "_GIT", (*coding_env._GIT, "-c", f"safe.directory={root}"))
    try:
        yield workspace
    finally:
        await workspace.stop()
