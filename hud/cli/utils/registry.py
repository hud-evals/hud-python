"""Local registry management for HUD environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hud.utils.hud_console import HUDConsole


def get_registry_dir() -> Path:
    """Get the base directory for the local HUD registry."""
    return Path.home() / ".hud" / "envs"


def extract_digest_from_image(image_ref: str) -> str:
    """Extract a digest identifier from a Docker image reference.

    Args:
        image_ref: Docker image reference (e.g., "image:tag@sha256:abc123...")

    Returns:
        Digest string for use as directory name (max 12 chars)
    """
    if "@sha256:" in image_ref:
        # Extract from digest reference: image@sha256:abc123...
        return image_ref.split("@sha256:")[-1][:12]
    elif image_ref.startswith("sha256:"):
        # Direct digest format
        return image_ref.split(":")[-1][:12]
    elif ":" in image_ref and "/" not in image_ref.split(":")[-1]:
        # Has a tag but no slashes after colon (not a port)
        tag = image_ref.split(":")[-1]
        return tag[:12] if tag else "latest"
    else:
        # No tag specified or complex format
        return "latest"


def extract_name_and_tag(image_ref: str) -> tuple[str, str]:
    """Extract organization/name and tag from Docker image reference.

    Args:
        image_ref: Docker image reference

    Returns:
        Tuple of (name, tag) where name includes org/repo

    Examples:
        docker.io/hudpython/test_init:latest@sha256:... -> (hudpython/test_init, latest)
        hudpython/myenv:v1.0 -> (hudpython/myenv, v1.0)
        myorg/myapp -> (myorg/myapp, latest)
    """
    # Remove digest if present
    if "@" in image_ref:
        image_ref = image_ref.split("@")[0]

    # Remove registry prefix if present
    if image_ref.startswith(("docker.io/", "registry-1.docker.io/", "index.docker.io/")):
        image_ref = "/".join(image_ref.split("/")[1:])

    # Extract tag
    if ":" in image_ref:
        name, tag = image_ref.rsplit(":", 1)
    else:
        name = image_ref
        tag = "latest"

    return name, tag


def save_to_registry(
    lock_data: dict[str, Any], image_ref: str, verbose: bool = False
) -> Path | None:
    """Save environment lock data to the local registry.

    Args:
        lock_data: The lock file data to save
        image_ref: Docker image reference for digest extraction
        verbose: Whether to show verbose output

    Returns:
        Path to the saved lock file, or None if save failed
    """
    hud_console = HUDConsole()

    try:
        # Extract digest for registry storage
        digest = extract_digest_from_image(image_ref)

        # Store under ~/.hud/envs/<digest>/
        local_env_dir = get_registry_dir() / digest
        local_env_dir.mkdir(parents=True, exist_ok=True)

        local_lock_path = local_env_dir / "hud.lock.yaml"
        with open(local_lock_path, "w") as f:
            yaml.dump(lock_data, f, default_flow_style=False, sort_keys=False)

        hud_console.success(f"Added to local registry: {digest}")
        if verbose:
            hud_console.info(f"Registry location: {local_lock_path}")

        return local_lock_path
    except Exception as e:
        if verbose:
            hud_console.warning(f"Failed to save to registry: {e}")
        return None


def load_from_registry(digest: str) -> dict[str, Any] | None:
    """Load environment lock data from the local registry.

    Args:
        digest: The digest/identifier of the environment

    Returns:
        Lock data dictionary, or None if not found
    """
    lock_path = get_registry_dir() / digest / "hud.lock.yaml"

    if not lock_path.exists():
        return None

    try:
        with open(lock_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def list_registry_entries() -> list[tuple[str, Path]]:
    """List all entries in the local registry.

    Returns:
        List of (digest, lock_path) tuples
    """
    registry_dir = get_registry_dir()

    if not registry_dir.exists():
        return []

    entries = []
    for digest_dir in registry_dir.iterdir():
        if not digest_dir.is_dir():
            continue

        lock_file = digest_dir / "hud.lock.yaml"
        if lock_file.exists():
            entries.append((digest_dir.name, lock_file))

    return entries
