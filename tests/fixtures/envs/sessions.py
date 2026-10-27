"""An actor that leaves its answer in its session's directory, and a verifier that reads it there.

Placed in two containers, a reward of 1 proves the runtime moved the actor's
session files into the verifier's container.
"""

from __future__ import annotations

from pathlib import Path

from hud import Environment
from hud.environment.env import current_session_id

env = Environment("sessions")


def session() -> Path:
    return Path("/runtime/sessions") / str(current_session_id.get())


@env.template()
async def solve(link: bool = False):
    answer = yield "Reply with: secret"
    directory = session()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "work.txt").write_text(str(answer))
    if link:
        (directory / "escape").symlink_to("/etc/passwd")
    yield {"score": 0.0, "answer": answer}


@env.template()
async def verify():
    yield ""
    work = session() / "work.txt"
    yield 1.0 if work.exists() and work.read_text() == "secret" else 0.0
