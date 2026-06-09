"""Docker helpers for the HUD CLI: daemon availability and per-env ``.env`` loading."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import TYPE_CHECKING

from .config import parse_env_file

if TYPE_CHECKING:
    from pathlib import Path


def load_env_vars_for_dir(env_dir: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from `<env_dir>/.env` if present.

    Returns an empty dict if no file is found or parsing fails.
    """
    env_file = env_dir / ".env"
    if not env_file.exists():
        return {}
    try:
        contents = env_file.read_text(encoding="utf-8")
        return parse_env_file(contents)
    except Exception:
        return {}


def _emit_docker_hints(error_text: str) -> None:
    """Parse common Docker connectivity errors and print platform-specific hints."""
    from hud.utils.hud_console import hud_console

    text = error_text.lower()
    system = platform.system()

    markers = [
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "error during connect",
        "permission denied while trying to connect",
        "no such file or directory",
        "pipe/dockerdesktop",
        "dockerdesktoplinuxengine",
        "//./pipe/docker",
        "/var/run/docker.sock",
    ]

    trimmed = error_text.strip()
    if len(trimmed) > 300:
        trimmed = trimmed[:300] + "..."

    if any(m in text for m in markers):
        hud_console.error("Docker does not appear to be running or accessible")
        if system == "Windows":
            hud_console.hint("Open Docker Desktop and wait until it shows 'Running'")
            hud_console.hint("If using WSL, enable integration for your distro in Docker Desktop")
        elif system == "Linux":
            hud_console.hint(
                "Start the daemon: sudo systemctl start docker (or service docker start)"
            )
            hud_console.hint("If permission denied: sudo usermod -aG docker $USER && re-login")
        elif system == "Darwin":
            hud_console.hint("Open Docker Desktop and wait until it shows 'Running'")
        else:
            hud_console.hint("Start Docker and ensure the daemon is reachable")
        hud_console.dim_info("Details", trimmed)
    else:
        hud_console.error("Docker returned an error")
        hud_console.dim_info("Details", trimmed)
        hud_console.hint("Is Docker running and accessible?")


def require_docker_running() -> None:
    """Ensure Docker CLI exists and daemon is reachable; print hints and exit if not."""
    import typer

    from hud.utils.hud_console import hud_console

    docker_path: str | None = shutil.which("docker")
    if not docker_path:
        hud_console.error("Docker CLI not found")
        hud_console.info("Install Docker Desktop (Windows/macOS) or Docker Engine (Linux)")
        hud_console.hint("After installation, start Docker and re-run this command")
        raise typer.Exit(1)

    try:
        result = subprocess.run(  # noqa: UP022
            [docker_path, "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
        if result.returncode == 0:
            return

        error_text = (result.stderr or "") + "\n" + (result.stdout or "")
        _emit_docker_hints(error_text)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        hud_console.error("Docker CLI not found on PATH")
        hud_console.hint("Install Docker and ensure 'docker' is on your PATH")
        raise typer.Exit(1) from e
    except subprocess.TimeoutExpired as e:
        hud_console.error("Docker did not respond in time")
        hud_console.hint(
            "Is Docker running? Open Docker Desktop and wait until it reports 'Running'"
        )
        raise typer.Exit(1) from e
    except typer.Exit:
        # Propagate cleanly without extra noise; hints already printed above
        raise
    except Exception:
        # Unknown failure - keep output minimal and avoid stack traces
        hud_console.hint("Is the Docker daemon running?")
        raise typer.Exit(1)  # noqa: B904
