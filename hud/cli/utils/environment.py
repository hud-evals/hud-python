"""Shared utilities for environment directory handling."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import toml

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def normalize_environment_name(name: str) -> str:
    """Normalize environment name to match SDK's Environment class.

    This ensures the name used in CLI matches what Environment.__init__()
    and the platform backend use, so scenario names are consistent.

    Rules:
    - Lowercase
    - Replace spaces and underscores with hyphens
    - Remove any non-alphanumeric chars except hyphens
    - Collapse multiple hyphens
    - Strip leading/trailing hyphens
    """
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or "environment"


def get_environment_name(
    directory: str | Path, name_override: str | None = None
) -> tuple[str, str]:
    """Resolve environment name with source tracking.

    Checks in order:
    1. Explicit override
    2. [tool.hud].name in pyproject.toml
    3. Directory name (sanitized)

    All names are normalized to match SDK's Environment class normalization,
    ensuring scenario prefixes are consistent between local and deployed envs.

    Returns:
        Tuple of (normalized_name, source) where source is "override", "config", or "auto"
    """
    if name_override:
        return normalize_environment_name(name_override), "override"

    # Check pyproject.toml for [tool.hud].name
    pyproject_path = Path(directory) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                config = toml.load(f)
            hud_config = config.get("tool", {}).get("hud", {})
            # Check for explicit name first
            if hud_config.get("name"):
                return normalize_environment_name(hud_config["name"]), "config"
            # Fall back to image name (without tag)
            if hud_config.get("image"):
                image = hud_config["image"]
                name = image.split(":")[0] if ":" in image else image
                # Remove org prefix if present
                if "/" in name:
                    name = name.split("/")[-1]
                return normalize_environment_name(name), "config"
        except Exception:  # noqa: S110
            pass

    # Auto-generate from directory name
    dir_path = Path(directory).resolve()
    dir_name = dir_path.name
    if not dir_name or dir_name == ".":
        dir_name = dir_path.parent.name
    return normalize_environment_name(dir_name), "auto"


def get_image_name(directory: str | Path, image_override: str | None = None) -> tuple[str, str]:
    """Resolve image name with source tracking.

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


def docker_build(directory: str | Path, image_name: str, no_cache: bool = False) -> bool:
    """Build Docker image for an environment (simple wrapper around ``docker build``).

    Returns:
        True if build succeeded, False otherwise
    """
    dir_path = Path(directory)

    # Validate directory exists and is a directory
    if not dir_path.exists():
        hud_console.error(f"Directory does not exist: {directory}")
        return False
    if not dir_path.is_dir():
        hud_console.error(f"Not a directory: {directory}")
        return False

    dockerfile_path = find_dockerfile(dir_path)
    if dockerfile_path is None:
        hud_console.error(f"No Dockerfile found in {directory}")
        hud_console.info("Expected: Dockerfile.hud or Dockerfile")
        return False

    build_cmd = ["docker", "build", "-t", image_name]

    # Specify the Dockerfile path if using Dockerfile.hud
    if dockerfile_path is not None and dockerfile_path.name != "Dockerfile":
        build_cmd.extend(["-f", str(dockerfile_path)])

    if no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(directory))

    hud_console.info(f"🔨 Building image: {image_name}{' (no cache)' if no_cache else ''}")
    if dockerfile_path is not None and dockerfile_path.name != "Dockerfile":
        hud_console.info(f"Using {dockerfile_path.name}")
    hud_console.info("")  # Empty line before Docker output

    # Just run Docker build directly - it has its own nice live display
    result = subprocess.run(build_cmd)

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
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],  # noqa: S607
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def find_dockerfile(directory: Path) -> Path | None:
    """Find Dockerfile in a directory, preferring Dockerfile.hud over Dockerfile."""
    hud_dockerfile = directory / "Dockerfile.hud"
    if hud_dockerfile.exists():
        return hud_dockerfile

    standard_dockerfile = directory / "Dockerfile"
    if standard_dockerfile.exists():
        return standard_dockerfile

    return None


def is_environment_directory(path: str | Path) -> bool:
    """Check if a path looks like an environment directory.

    An environment directory should have:
    - A Dockerfile (Dockerfile.hud or Dockerfile)
    - A pyproject.toml file
    - Optionally a src directory
    """
    dir_path = Path(path)
    if not dir_path.exists():
        return False
    if not dir_path.is_dir():
        return False

    # Must have Dockerfile.hud or Dockerfile
    if find_dockerfile(dir_path) is None:
        return False

    # Must have pyproject.toml
    return (dir_path / "pyproject.toml").exists()
