"""The user's machine, as the SDK under test meets it: its environment variables, the
``hud`` command and ``uv`` it runs, the files it writes, and the spans the SDK leaves
on its disk.

:class:`HudEnv` is the only place tests touch ``hud.settings``: the SDK reads its
configuration from environment variables once, when ``hud.settings`` is imported, so
:class:`HudEnv` sets them for this process and every subprocess it starts and rebuilds
the settings object from them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.settings import Settings, settings
from hud.utils.gateway import list_gateway_models

from .services import SERVICE_ENV

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest


# Every service the SDK calls points at a closed local port until a test opts into
# the fake services, so nothing a test does can reach a real HUD deployment. The
# web app is only ever printed as a link, so it keeps its default.
CLOSED_PORT = "http://127.0.0.1:9"

ISOLATED = {
    "HUD_TELEMETRY_ENABLED": "0",
    "HUD_CLI_ANALYTICS_ENABLED": "0",
    "HUD_SKIP_VERSION_CHECK": "1",
    **{var: f"{CLOSED_PORT}/{service}" for service, var in SERVICE_ENV.items() if service != "web"},
}

# Provider SDKs read these directly, bypassing hud.settings.
PROVIDER_ENV = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


def setting_variables() -> list[str]:
    """Environment variable names the SDK settings read."""
    return [
        field.validation_alias
        for field in Settings.model_fields.values()
        if isinstance(field.validation_alias, str)
    ]


class HudEnv:
    """Isolated SDK configuration for one test: a fresh home and no developer credentials."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
        self._monkeypatch = monkeypatch
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        # The Docker CLI finds its plugins (compose) and current context under its
        # config directory; keep the real one so a real daemon stays reachable.
        monkeypatch.setenv(
            "DOCKER_CONFIG", os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker")
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        for name in (*setting_variables(), *PROVIDER_ENV):
            monkeypatch.delenv(name, raising=False)
        for name, value in ISOLATED.items():
            monkeypatch.setenv(name, value)
        self._apply()

    def set(self, **values: str | None) -> None:
        """Set variables (``None`` unsets one) and re-read the SDK settings from them."""
        for name, value in values.items():
            if value is None:
                self._monkeypatch.delenv(name, raising=False)
            else:
                self._monkeypatch.setenv(name, value)
        self._apply()

    def get(self, name: str) -> str | None:
        return os.environ.get(name)

    def _apply(self) -> None:
        fresh = Settings(_env_file=None)
        for name in Settings.model_fields:
            self._monkeypatch.setattr(settings, name, getattr(fresh, name))
        # The gateway model catalog is cached per process; a new configuration
        # may point at a different catalog.
        list_gateway_models.cache_clear()


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


BOX_DRAWING = re.compile(r"[\u2500-\u259f]+")


def words(text: str) -> list[str]:
    """``text`` as a reader sees it: box drawing and padding gone, blank lines dropped.

    Rich lays panels out for the terminal's width, so output is compared by its words.
    """
    stripped = (" ".join(BOX_DRAWING.sub(" ", line).split()) for line in text.splitlines())
    return [line for line in stripped if line]


@dataclass(frozen=True)
class Result:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def json(self) -> Any:
        """The single JSON document a ``--json`` command prints on stdout."""
        return json.loads(self.stdout)

    @property
    def lines(self) -> list[str]:
        """Stdout as a reader sees its words; see :func:`words`."""
        return words(self.stdout)

    def __str__(self) -> str:
        return f"exit {self.exit_code}\n--- stdout\n{self.stdout}\n--- stderr\n{self.stderr}"


class Hud:
    """``hud`` bound to a working directory and a terminal width. Use the ``hud`` fixture.

    The subprocess inherits this process's environment, which the ``hud_env``
    fixture has already isolated: a temporary home, no developer credentials,
    and every service URL pointed at the fake services or a closed port.
    ``python`` runs the CLI from another interpreter, such as a project's own
    virtual environment; it defaults to this one.
    """

    def __init__(self, cwd: Path, *, python: Path | None = None, columns: int = 120) -> None:
        self.cwd = cwd
        self.python = str(python) if python is not None else sys.executable
        self.columns = columns
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
            env=self._env(env),
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return Result(completed.returncode, completed.stdout, completed.stderr)

    def start(
        self, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> subprocess.Popen[str]:
        """Start ``hud`` without waiting, for commands that run until signalled.

        Stdout and stderr are pipes; the caller stops the process and reads them.
        """
        return subprocess.Popen(
            [self.python, "-m", "hud.cli", *args],
            cwd=cwd or self.cwd,
            env=self._env(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _env(self, extra: dict[str, str] | None) -> dict[str, str]:
        terminal = {"COLUMNS": str(self.columns), "NO_COLOR": "1", "TERM": "dumb"}
        return {**os.environ, **terminal, **(extra or {})}


def scrub(text: str, *paths: Path) -> str:
    """Replace temporary paths and UUIDs so output compares across runs."""
    for index, path in enumerate(paths):
        text = text.replace(str(path), f"<path{index}>")
    return UUID.sub("<uuid>", text)


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


def write_files(root: Path, files: Mapping[str, bytes | str]) -> None:
    """Write ``files`` under ``root``; a name ending in ``/`` is an empty directory."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        if relative.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)


STEP_SCHEMA = "hud.step.v1"
ROBOT_STEP_SCHEMA = "hud.robot.step.v1"


def spans(trace_id: str | None, span_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every span exported for the run ``trace_id``, in start order.

    Spans land in ``<span_dir>/<trace id as 32 hex>.jsonl``; the span directory
    is the SDK's own (``HUD_TELEMETRY_LOCAL_DIR``, else ``~/.hud/spans`` while
    uploads are off).
    """
    if trace_id is None:
        raise ValueError("the run has no trace id, so it exported no spans")
    directory = span_dir or Path(settings.span_dir or "")
    path = directory / f"{uuid.UUID(trace_id).hex}.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return sorted(records, key=lambda span: span.get("start_time", ""))


def steps(
    trace_id: str | None, span_dir: Path | None = None, *, schema: str = STEP_SCHEMA
) -> list[dict[str, Any]]:
    """The step payloads of a run, in order; ``schema=ROBOT_STEP_SCHEMA`` for robot telemetry."""
    return [
        span["attributes"]["hud.payload"]
        for span in spans(trace_id, span_dir)
        if span.get("attributes", {}).get("hud.schema") == schema
    ]
