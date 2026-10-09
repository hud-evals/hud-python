"""A token only the workspace holds: the agent can answer only by reading it through a shell."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
from pathlib import Path

from hud import Environment

ROOT = Path(tempfile.gettempdir()) / f"hud-oracle-secret-{os.getpid()}"

env = Environment("oracle-secret")
env.workspace(ROOT, guest_path=str(ROOT))


@env.shutdown
async def remove_workspace() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)


@env.template(id="read-secret")
async def read_secret():
    token = secrets.token_hex(8)
    (ROOT / "secret.txt").write_text(token, encoding="utf-8")
    answer = yield f"Read the file {ROOT / 'secret.txt'} and reply with its contents only."
    yield 1.0 if (answer or "").strip() == token else 0.0


tasks = [read_secret()]
