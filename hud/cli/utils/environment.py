"""Shared utilities for environment directory handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

import toml

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def get_image_name(directory: str | Path, image_override: str | None = None) -> tuple[str, str]:
    """
    Resolve image name with source tracking.

    Returns:
        Tuple of (image_name, source) where source is "override", "cache", or "auto"
    """
    if image_override:
        return image_override, "override"

    # Check pyproject.toml
    pyproject_path = Path(directory) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                config = toml.load(f)
            if config.get("tool", {}).get("hud", {}).get("image"):
                return config["tool"]["hud"]["image"], "cache"
        except Exception:
            hud_console.error("Error loading pyproject.toml")

    # Auto-generate with :dev tag (replace underscores with hyphens)
    dir_path = Path(directory).resolve()  # Get absolute path first
    dir_name = dir_path.name
    if not dir_name or dir_name == ".":
        # If we're in root or have empty name, use parent directory
        dir_name = dir_path.parent.name
    # Replace underscores with hyphens for Docker image names
    dir_name = dir_name.replace("_", "-")
    return f"{dir_name}:dev", "auto"


def update_pyproject_toml(directory: str | Path, image_name: str, silent: bool = False) -> None:
    """Update pyproject.toml with image name."""
    pyproject_path = Path(directory) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                config = toml.load(f)

            # Ensure [tool.hud] exists
            if "tool" not in config:
                config["tool"] = {}
            if "hud" not in config["tool"]:
                config["tool"]["hud"] = {}

            # Update image name
            config["tool"]["hud"]["image"] = image_name

            # Write back
            with open(pyproject_path, "w") as f:
                toml.dump(config, f)

            if not silent:
                hud_console.success(f"Updated pyproject.toml with image: {image_name}")
        except Exception as e:
            if not silent:
                hud_console.warning(f"Could not update pyproject.toml: {e}")


def build_environment(directory: str | Path, image_name: str, no_cache: bool = False) -> bool:
    """Build Docker image for an environment.

    Returns:
        True if build succeeded, False otherwise
    """
    build_cmd = ["docker", "build", "-t", image_name]
    if no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(directory))

    hud_console.info(f"🔨 Building image: {image_name}{' (no cache)' if no_cache else ''}")
    hud_console.info("")  # Empty line before Docker output

    # Just run Docker build directly - it has its own nice live display
    result = subprocess.run(build_cmd)  # noqa: S603

    if result.returncode == 0:
        hud_console.info("")  # Empty line after Docker output
        hud_console.success(f"Build successful! Image: {image_name}")
        # Update pyproject.toml (silently since we already showed success)
        update_pyproject_toml(directory, image_name, silent=True)
        return True
    else:
        hud_console.error("Build failed!")
        return False


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", image_name],  # noqa: S607
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def is_environment_directory(path: str | Path) -> bool:
    """Check if a path looks like an environment directory.

    An environment directory should have:
    - A Dockerfile
    - A pyproject.toml file
    - Optionally a src directory
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        return False

    # Must have Dockerfile
    if not (dir_path / "Dockerfile").exists():
        return False

    # Must have pyproject.toml
    return (dir_path / "pyproject.toml").exists()
