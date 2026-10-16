"""``uv`` as the developer or CI runs it, whatever ``HOME`` a test isolated."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class Uv:
    """``uv`` with the developer's cache and Python installs, whatever ``HOME`` a test set."""

    def __init__(self, real_environment: dict[str, str]) -> None:
        executable = shutil.which("uv", path=real_environment.get("PATH"))
        if executable is None:
            raise RuntimeError("the template scenarios need uv on PATH")
        self.executable = executable
        self.env = {
            "UV_CACHE_DIR": self.ask(real_environment, "cache", "dir"),
            "UV_PYTHON_INSTALL_DIR": self.ask(real_environment, "python", "dir"),
        }

    def ask(self, environment: dict[str, str], *args: str) -> str:
        return subprocess.run(
            [self.executable, *args], env=environment, capture_output=True, text=True, check=True
        ).stdout.strip()

    def __call__(self, *args: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        completed = subprocess.run(
            [self.executable, *args],
            cwd=cwd,
            env={**os.environ, **self.env, **(env or {})},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"uv {' '.join(args)}\n{completed.stderr}"
