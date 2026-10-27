"""The user's machine as the SDK under test meets it, and the fixtures that wire it in.

Every test runs against an isolated SDK configuration: a temporary home, no
developer credentials, telemetry uploads off, and every HUD service URL pointed
at a closed local port. :class:`HudEnv` is the only place tests touch
``hud.settings``: the SDK reads its configuration from environment variables
once, when ``hud.settings`` is imported, so :class:`HudEnv` sets them for this
process and every subprocess it starts and rebuilds the settings object from
them. A test that talks to HUD services asks for the
``services`` fixture, which points the SDK at the fake backend instead.

Tests marked ``e2e`` need something outside the process (a Docker daemon, a
Linux sandbox, a real HUD account) and only run when selected with ``-m``;
they skip themselves when what they need is missing.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dotenv import dotenv_values

from hud.settings import Settings, settings
from hud.telemetry import flush
from hud.utils.gateway import list_gateway_models
from tests.fixtures.models import Models
from tests.fixtures.services import SERVICE_ENV, FakeServices

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
# The fixture environments that need nothing but the SDK, so a container can serve them.
IMAGE_FIXTURES = ("echo", "faulty", "sessions", "broken", "stalls")

# The developer's or CI's own environment, captured before any test isolates it.
# Only ``live`` tests read credentials from it.
REAL_ENVIRONMENT = dict(os.environ)

SERVICE_URL_VARIABLES = (
    "HUD_API_URL",
    "HUD_TELEMETRY_URL",
    "HUD_GATEWAY_URL",
    "HUD_RUNTIME_URL",
    "HUD_RL_URL",
    "HUD_WEB_URL",
)


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
        settings_read = [
            field.validation_alias
            for field in Settings.model_fields.values()
            if isinstance(field.validation_alias, str)
        ]
        for name in (*settings_read, *PROVIDER_ENV):
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


async def eventually(
    condition: Callable[[], bool | Awaitable[bool]], *, within: float = 10.0
) -> None:
    """Poll until ``condition()`` holds; raise ``TimeoutError`` after ``within`` seconds.

    ``condition`` may be a coroutine function, for a check that has to await.
    """
    async with asyncio.timeout(within):
        while True:
            held = condition()
            if await held if inspect.isawaitable(held) else held:
                return
            await asyncio.sleep(0.01)


def real_hud_api_key() -> str | None:
    """``HUD_API_KEY`` from the real environment or the real ``~/.hud/.env``."""
    if key := REAL_ENVIRONMENT.get("HUD_API_KEY"):
        return key
    home = REAL_ENVIRONMENT.get("HOME")
    if home is None:
        return None
    return dotenv_values(Path(home) / ".hud" / ".env").get("HUD_API_KEY")


@functools.cache
def real_modal_token() -> tuple[str, str] | None:
    """The token Modal resolves from the real environment and config file, asked of Modal
    in a process of its own so the tests' isolated home never reaches it."""
    if importlib.util.find_spec("modal") is None:
        return None
    reads = (
        "import json; from modal.config import config; "
        "print(json.dumps([config.get('token_id'), config.get('token_secret')]))"
    )
    resolved = subprocess.run(
        [sys.executable, "-c", reads],
        env=REAL_ENVIRONMENT,
        capture_output=True,
        text=True,
        check=True,
    )
    token_id, token_secret = json.loads(resolved.stdout)
    return (token_id, token_secret) if token_id and token_secret else None


@pytest.fixture(autouse=True)
def hud_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HudEnv:
    """The SDK's configuration for this test; change it with ``hud_env.set(...)``."""
    return HudEnv(monkeypatch, tmp_path)


@pytest.fixture(scope="session")
def services_server() -> Iterator[FakeServices]:
    server = FakeServices()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def services(services_server: FakeServices, hud_env: HudEnv) -> FakeServices:
    """The fake HUD backend, with the SDK configured to reach it and no routes declared."""
    services_server.reset()
    hud_env.set(**services_server.env())
    return services_server


@pytest.fixture
def reporting(services: FakeServices, hud_env: HudEnv) -> Iterator[FakeServices]:
    """The HUD services accepting job and trace reports and span uploads, with telemetry on
    and a key.

    Span uploads drain before the test ends, so none reaches a later test's services.
    """
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    for path in ("/v2/trace/job/{id}/enter", "/v2/trace/{id}/enter", "/v2/trace/{id}/exit"):
        services.route("api", "POST", path, json={})
    services.route("telemetry", "POST", "/trace/{id}/telemetry-upload", json={})
    yield services
    assert flush(timeout=10)


@pytest.fixture
def models(services: FakeServices) -> Models:
    """Scripted model providers behind the fake gateway."""
    return Models(services)


@pytest.fixture
def hud(tmp_path: Path) -> Hud:
    """The real ``hud`` CLI, run in a subprocess from ``tmp_path / "work"``."""
    return Hud(tmp_path / "work")


@pytest.fixture(scope="session")
def uv() -> Uv:
    return Uv(REAL_ENVIRONMENT)


@pytest.fixture(scope="session")
def sdk_wheel(uv: Uv, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This checkout's SDK as a wheel, built once per test worker."""
    out = tmp_path_factory.mktemp("sdk-wheel")
    uv("build", "--wheel", "--no-create-gitignore", "--out-dir", str(out), str(ROOT), cwd=out)
    (wheel,) = out.glob("hud-*.whl")
    return wheel


@pytest.fixture(scope="session")
def fixture_image(
    sdk_wheel: Path, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], str]:
    """The image serving fixture environment ``name``, built on first use; returns its tag."""
    context = tmp_path_factory.mktemp("fixture-image")
    shutil.copy(sdk_wheel, context)
    shutil.copy(FIXTURES / "Dockerfile", context)
    (context / "envs").mkdir()
    for name in IMAGE_FIXTURES:
        shutil.copy(FIXTURES / "envs" / f"{name}.py", context / "envs")
    built: dict[str, str] = {}

    def image(name: str) -> str:
        if name not in built:
            tag = f"hud-test-fixture:{name}"
            subprocess.run(
                ["docker", "build", "--quiet", "--build-arg", f"FIXTURE={name}", "--tag", tag, "."],
                cwd=context,
                env=REAL_ENVIRONMENT,
                capture_output=True,
                check=True,
            )
            built[name] = tag
        return built[name]

    return image


@pytest.fixture(scope="session")
def modal_image(sdk_wheel: Path) -> Callable[[str], Any]:
    """A Modal image serving fixture environment ``name`` as ``/hud/env.py``, from this
    checkout's SDK; Modal builds it on first use and caches it after."""
    import modal

    def image(name: str) -> Any:
        return (
            modal.Image.debian_slim(python_version="3.12")
            .add_local_file(sdk_wheel, f"/tmp/{sdk_wheel.name}", copy=True)
            .run_commands(f"pip install /tmp/{sdk_wheel.name}")
            .add_local_file(FIXTURES / "envs" / f"{name}.py", "/hud/env.py", copy=True)
        )

    return image


@pytest.fixture
def modal_account(hud_env: HudEnv) -> HudEnv:
    """Hand Modal the real account the developer or CI configured. Modal reads its config
    file once, when first imported, so the token goes in as variables."""
    token = real_modal_token()
    assert token is not None
    hud_env.set(MODAL_TOKEN_ID=token[0], MODAL_TOKEN_SECRET=token[1])
    return hud_env


@pytest.fixture
def live(hud_env: HudEnv) -> HudEnv:
    """Configure the SDK for a real HUD account: the real key and the real services."""
    key = real_hud_api_key()
    if key is None:
        pytest.skip("needs HUD_API_KEY")
    hud_env.set(
        HUD_API_KEY=key,
        **{name: REAL_ENVIRONMENT.get(name) for name in SERVICE_URL_VARIABLES},
    )
    return hud_env


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=20)
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip selected ``e2e`` tests whose daemon, sandbox, or credentials are missing."""
    del config
    checks = {
        "docker": (_docker_available, "needs a running Docker daemon"),
        "sandbox": (
            lambda: sys.platform == "linux" and os.geteuid() == 0 and bool(shutil.which("bwrap")),
            "needs Linux, root, and bubblewrap",
        ),
        "live": (lambda: real_hud_api_key() is not None, "needs HUD_API_KEY"),
        "modal": (lambda: real_modal_token() is not None, "needs a Modal account"),
        "lerobot": (
            lambda: importlib.util.find_spec("lerobot") is not None,
            "needs lerobot[dataset]",
        ),
    }
    available: dict[str, bool] = {}
    for item in items:
        for mark, (check, reason) in checks.items():
            if mark not in item.keywords:
                continue
            if mark not in available:
                available[mark] = check()
            if not available[mark]:
                item.add_marker(pytest.mark.skip(reason=reason))
