"""Run the real ``hud`` CLI in a subprocess, the way a user does."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@dataclass(frozen=True)
class Result:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def json(self) -> Any:
        """The single JSON document a ``--json`` command prints on stdout."""
        return json.loads(self.stdout)

    def __str__(self) -> str:
        return f"exit {self.exit_code}\n--- stdout\n{self.stdout}\n--- stderr\n{self.stderr}"


class Hud:
    """``hud`` bound to a working directory. Use the ``hud`` fixture.

    The subprocess inherits this process's environment, which the ``hud_env``
    fixture has already isolated: a temporary home, no developer credentials,
    and every service URL pointed at the fake services or a closed port.
    ``python`` runs the CLI from another interpreter, such as a project's own
    virtual environment; it defaults to this one.
    """

    def __init__(self, cwd: Path, *, python: Path | None = None) -> None:
        self.cwd = cwd
        self.python = str(python) if python is not None else sys.executable
        cwd.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        *args: str,
        cwd: Path | None = None,
        input: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 120,
    ) -> Result:
        completed = subprocess.run(
            [self.python, "-m", "hud.cli", *args],
            cwd=cwd or self.cwd,
            env={**os.environ, "COLUMNS": "120", "NO_COLOR": "1", "TERM": "dumb", **(env or {})},
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return Result(completed.returncode, completed.stdout, completed.stderr)


def scrub(text: str, *paths: Path) -> str:
    """Replace temporary paths and UUIDs so output compares across runs."""
    for index, path in enumerate(paths):
        text = text.replace(str(path), f"<path{index}>")
    return UUID.sub("<uuid>", text)
