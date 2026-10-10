"""Generated environment projects: ``hud init`` output with the SDK from this checkout installed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.harness import Hud

from .lifecycle import assert_clean_lifecycle

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = sorted(
    path.name for path in (ROOT / "environments").iterdir() if (path / "env.py").is_file()
)
ROWS = "ci-tasks.json"
CHECKS = (
    ("-m", "ruff", "format", ".", "--check"),
    ("-m", "ruff", "check", "."),
    ("-m", "pytest", "-q", "-p", "no:cacheprovider"),
)
EXPORT_TASKS = (
    "from hud.eval import Taskset; Taskset.from_file('tasks.py').to_file('.hud/tasks.json')"
)


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


def system_python() -> str:
    """A CPython under /usr: inside the workspace sandbox, only /usr is visible to the grader."""
    for version in ("3.12", "3.11"):
        if found := shutil.which(f"python{version}", path="/usr/local/bin:/usr/bin"):
            return found
    raise RuntimeError("needs Python 3.11 or 3.12 under /usr/local/bin or /usr/bin")


@dataclass(frozen=True)
class Project:
    """A generated project and its own ``hud``, run from the project's environment."""

    root: Path
    python: Path

    @property
    def hud(self) -> Hud:
        return Hud(self.root, python=self.python)

    def run(self, *args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
        """Run the project's Python with ``args``, the way ``uv run python`` would."""
        return subprocess.run(
            [str(self.python), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def check(self) -> None:
        """The template's own lint and tests, as a user runs them before deploying."""
        for args in CHECKS:
            completed = self.run(*args)
            assert completed.returncode == 0, (
                f"{' '.join(args)}\n{completed.stdout}\n{completed.stderr}"
            )

    def bundle_sdk_into_image(self) -> None:
        """Copy the bundled SDK into the image ahead of its ``uv sync``, for a deploy to build."""
        dockerfile = self.root / "Dockerfile.hud"
        lines = dockerfile.read_text().splitlines(keepends=True)
        sync = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("RUN") and "uv sync" in line
        )
        lines.insert(sync, "COPY .hud-sdk .hud-sdk\n")
        dockerfile.write_text("".join(lines))
        with (self.root / ".dockerignore").open("a") as ignored:
            ignored.write("!.hud-sdk/\n!.hud-sdk/**\n")

    def write_rows(self, data_file: str | None) -> Path:
        """Write the first task as a JSON row beside the env source, attaching ``data_file``."""
        exported = self.run("-c", EXPORT_TASKS)
        assert exported.returncode == 0, exported.stderr
        row = json.loads((self.root / ".hud" / "tasks.json").read_text())[0]
        if data_file is not None:
            row["args"]["attachments"] = [{"file_id": data_file, "path": "notes.txt"}]
            row["args"]["prompt"] = "Read files/notes.txt first. " + row["args"]["prompt"]
        path = self.root / ROWS
        path.write_text(json.dumps([row], indent=2))
        return path


def generate(
    hud: Hud,
    template: str,
    directory: Path,
    *,
    wheel: Path,
    uv: Uv,
    environment: Path | None = None,
) -> Project:
    """``hud init`` a project from ``template``, then install this checkout's SDK into it.

    The wheel is copied into the project, as a deploy needs it inside the build
    context. ``environment`` installs the project's Python somewhere other than
    ``.venv``, the way an image does.
    """
    created = hud("init", directory.name, "--dir", str(directory.parent), "--template", template)
    assert created.exit_code == 0, created
    bundled = directory / ".hud-sdk" / wheel.name
    bundled.parent.mkdir()
    shutil.copy(wheel, bundled)
    uv("add", "--no-sync", f".hud-sdk/{wheel.name}", cwd=directory)
    placement = {}
    if environment is not None:
        placement = {"UV_PROJECT_ENVIRONMENT": str(environment), "UV_PYTHON": system_python()}
    uv("sync", "--all-extras", cwd=directory, env=placement)
    python = (environment or directory / ".venv") / "bin" / "python"
    return Project(directory, python)


def assert_lifecycle(
    run: dict[str, Any], recorded: list[dict[str, Any]], data_file: str | None
) -> dict[str, Any]:
    """A clean single-run lifecycle; a declared data file reaches the start frame."""
    assert run["is_error"] is False, run
    setup, grade = assert_clean_lifecycle(recorded)
    if data_file is not None:
        assert setup["data_files"] == [{"path": "files/notes.txt", "file_id": data_file}]
    return grade
