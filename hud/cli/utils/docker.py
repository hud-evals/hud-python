"""Docker utilities for HUD CLI."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess


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
    except Exception as e:
        hud_console.error(f"Docker check failed: {e}")
        hud_console.hint("Is the Docker daemon running?")
        raise typer.Exit(1) from e
