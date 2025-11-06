"""Docker utilities for HUD CLI.

This module centralizes helpers for constructing Docker commands and
standardizes environment variable handling for "folder mode" (environment
directories that include a `.env` file and/or `hud.lock.yaml`).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .config import parse_env_file

# Note: we deliberately avoid the stricter is_environment_directory() check here
# to allow folder mode with only a Dockerfile or only a pyproject.toml.


def get_docker_cmd(image: str) -> list[str] | None:
    """
    Extract the CMD from a Docker image.

    Args:
        image: Docker image name

    Returns:
        List of command parts or None if not found
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["docker", "inspect", image],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )

        inspect_data = json.loads(result.stdout)
        if inspect_data and len(inspect_data) > 0 and isinstance(inspect_data[0], dict):
            config = inspect_data[0].get("Config", {})
            cmd = config.get("Cmd", [])
            return cmd if cmd else None

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", image_name],  # noqa: S607
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def remove_container(container_name: str) -> bool:
    """Remove a Docker container by name.

    Args:
        container_name: Name of the container to remove

    Returns:
        True if successful or container doesn't exist, False on error
    """
    try:
        subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container_name],  # noqa: S607
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,  # Don't raise error if container doesn't exist
        )
        return True
    except Exception:
        return False


def generate_container_name(identifier: str, prefix: str = "hud") -> str:
    """Generate a safe container name from an identifier.

    Args:
        identifier: Image name or other identifier
        prefix: Prefix for the container name

    Returns:
        Safe container name with special characters replaced
    """
    # Replace special characters with hyphens
    safe_name = identifier.replace(":", "-").replace("/", "-").replace("\\", "-")
    return f"{prefix}-{safe_name}"


def build_run_command(image: str, docker_args: list[str] | None = None) -> list[str]:
    """Construct a standard docker run command used across CLI commands.

    Args:
        image: Docker image name to run
        docker_args: Additional docker args to pass before the image

    Returns:
        The docker run command list
    """
    args = docker_args or []
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        *args,
        image,
    ]


def detect_environment_dir(start_dir: Path | None = None) -> Path | None:
    """Detect an environment directory for folder mode.

    Detection order:
    - Current directory containing `hud.lock.yaml`
    - Parent directory containing `hud.lock.yaml`
    - Current directory that looks like an environment if it has either a
      `Dockerfile` or a `pyproject.toml` (looser than `is_environment_directory`).

    Returns the detected directory path or None if not found.
    """
    base = (start_dir or Path.cwd()).resolve()

    # Check current then parent for lock file
    for candidate in [base, base.parent]:
        if (candidate / "hud.lock.yaml").exists():
            return candidate

    # Fallback: treat as env if it has Dockerfile OR pyproject.toml
    if (base / "Dockerfile").exists() or (base / "pyproject.toml").exists():
        return base

    return None


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


def build_env_flags(env_vars: dict[str, str]) -> list[str]:
    """Convert an env dict into a flat list of `-e KEY=VALUE` flags."""
    flags: list[str] = []
    for key, value in env_vars.items():
        flags.extend(["-e", f"{key}={value}"])
    return flags


def create_docker_run_command(
    image: str,
    docker_args: list[str] | None = None,
    env_dir: Path | str | None = None,
    extra_env: dict[str, str] | None = None,
    name: str | None = None,
    interactive: bool = True,
    remove: bool = True,
) -> list[str]:
    """Create a standardized `docker run` command with folder-mode envs.

    - If `env_dir` is provided (or auto-detected), `.env` entries are injected as
      `-e KEY=VALUE` flags before the image.
    - `extra_env` allows callers to provide additional env pairs that override
      variables from `.env`.

    Args:
        image: Docker image to run
        docker_args: Additional docker args (volumes, ports, etc.)
        env_dir: Environment directory to load `.env` from; if None, auto-detect
        extra_env: Additional env variables to inject (takes precedence)
        name: Optional container name
        interactive: Include `-i` flag (default True)
        remove: Include `--rm` flag (default True)

    Returns:
        Fully constructed docker run command
    """
    cmd: list[str] = ["docker", "run"]
    if remove:
        cmd.append("--rm")
    if interactive:
        cmd.append("-i")
    if name:
        cmd.extend(["--name", name])

    # Load env from `.env` in detected env directory
    env_dir_path: Path | None = (
        Path(env_dir).resolve() if isinstance(env_dir, (str, Path)) else detect_environment_dir()
    )

    merged_env: dict[str, str] = {}
    if env_dir_path is not None:
        merged_env.update(load_env_vars_for_dir(env_dir_path))
    if extra_env:
        # Caller-provided values override .env
        merged_env.update(extra_env)

    # Insert env flags before other args
    if merged_env:
        cmd.extend(build_env_flags(merged_env))

    # Add remaining args (volumes, ports, etc.)
    if docker_args:
        cmd.extend(docker_args)

    cmd.append(image)
    return cmd


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
        trimmed = error_text.strip()
        if len(trimmed) > 300:
            trimmed = trimmed[:300] + "..."
        hud_console.dim_info("Details", trimmed)
    else:
        from hud.utils.hud_console import hud_console as _hc

        _hc.error("Docker returned an error")
        trimmed = error_text.strip()
        if len(trimmed) > 300:
            trimmed = trimmed[:300] + "..."
        _hc.dim_info("Details", trimmed)
        _hc.hint("Is Docker running and accessible?")


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
        result = subprocess.run(  # noqa: UP022, S603
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
