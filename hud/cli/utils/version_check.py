"""Version checking utilities for HUD CLI.

This module handles checking for updates to the hud-python package
and prompting users to upgrade when a new version is available.

Features:
- Checks PyPI for the latest version of hud-python
- Caches results for 6 hours to avoid excessive API calls
- Displays a friendly prompt when an update is available
- Can be disabled with HUD_SKIP_VERSION_CHECK=1 environment variable

The version check runs automatically at the start of most CLI commands,
but is skipped for help and version commands to keep them fast.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from packaging import version

from hud.utils.hud_console import HUDConsole

# Logger for version checking
logger = logging.getLogger(__name__)

# Cache location for version check data
CACHE_DIR = Path.home() / ".hud" / ".cache"
VERSION_CACHE_FILE = CACHE_DIR / "version_check.json"

# Cache duration in seconds (6 hours)
CACHE_DURATION = 6 * 60 * 60

# PyPI API URL for package info
PYPI_URL = "https://pypi.org/pypi/hud-python/json"


class VersionInfo(NamedTuple):
    """Version information from PyPI."""

    latest: str
    current: str
    is_outdated: bool
    checked_at: float


def _get_current_version() -> str:
    """Get the currently installed version of hud-python."""
    try:
        from hud import __version__

        return __version__
    except ImportError:
        return "unknown"


def _fetch_latest_version() -> str | None:
    """Fetch the latest version from PyPI.

    Returns:
        The latest version string, or None if the request fails.
    """
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(PYPI_URL)
            if response.status_code == 200:
                data = response.json()
                return data["info"]["version"]
    except Exception:  # noqa: S110
        # Silently fail - we don't want to disrupt the user's workflow
        # if PyPI is down or there's a network issue
        pass
    return None


def _load_cache() -> VersionInfo | None:
    """Load cached version information.

    Returns:
        Cached VersionInfo if valid, None otherwise.
    """
    if not VERSION_CACHE_FILE.exists():
        return None

    try:
        with open(VERSION_CACHE_FILE) as f:
            data = json.load(f)

        # Check if cache is still valid
        if time.time() - data["checked_at"] > CACHE_DURATION:
            return None

        return VersionInfo(
            latest=data["latest"],
            current=data["current"],
            is_outdated=data["is_outdated"],
            checked_at=data["checked_at"],
        )
    except Exception:
        # If cache is corrupted, return None
        return None


def _save_cache(info: VersionInfo) -> None:
    """Save version information to cache.

    Args:
        info: Version information to cache.
    """
    try:
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(VERSION_CACHE_FILE, "w") as f:
            json.dump(
                {
                    "latest": info.latest,
                    "current": info.current,
                    "is_outdated": info.is_outdated,
                    "checked_at": info.checked_at,
                },
                f,
            )
    except Exception:  # noqa: S110
        # Silently fail if we can't write cache
        pass


def _compare_versions(current: str, latest: str) -> bool:
    """Compare versions to determine if an update is available.

    Args:
        current: Current version string
        latest: Latest version string

    Returns:
        True if latest is newer than current, False otherwise.
    """
    if current == "unknown":
        return False

    try:
        current_v = version.parse(current)
        latest_v = version.parse(latest)
        return latest_v > current_v
    except Exception:
        # If we can't parse versions, assume no update needed
        return False


def check_for_updates() -> VersionInfo | None:
    """Check for updates to hud-python.

    This function checks PyPI for the latest version and caches the result
    for 6 hours to avoid excessive API calls.

    Returns:
        VersionInfo if check succeeds, None otherwise.
    """
    # Check if we're in CI/testing environment
    if os.environ.get("CI") or os.environ.get("HUD_SKIP_VERSION_CHECK"):
        return None

    # Get current version first
    current = _get_current_version()
    if current == "unknown":
        return None

    # Try to load from cache
    cached_info = _load_cache()

    # If cache exists but current version has changed (user upgraded), invalidate cache
    if cached_info and cached_info.current != current:
        cached_info = None  # Force fresh check

    if cached_info:
        # Update the current version in the cached info to reflect reality
        # but keep the cached latest version and timestamp
        return VersionInfo(
            latest=cached_info.latest,
            current=current,  # Use actual current version, not cached
            is_outdated=_compare_versions(current, cached_info.latest),
            checked_at=cached_info.checked_at,
        )

    # Fetch latest version from PyPI
    latest = _fetch_latest_version()
    if not latest:
        return None

    # Compare versions
    is_outdated = _compare_versions(current, latest)

    # Create version info
    info = VersionInfo(
        latest=latest,
        current=current,
        is_outdated=is_outdated,
        checked_at=time.time(),
    )

    # Save to cache
    _save_cache(info)

    return info


def display_update_prompt(console: HUDConsole | None = None) -> None:
    """Display update prompt if a new version is available.

    This function checks for updates and displays a prompt to the user
    if their version is outdated.

    Args:
        console: HUDConsole instance for output. If None, creates a new one.
    """
    if console is None:
        console = HUDConsole(logger=logger)

    try:
        info = check_for_updates()
        if info and info.is_outdated:
            # Create update message
            update_msg = (
                f"🆕 A new version of hud-python is available: "
                f"[bold cyan]{info.latest}[/bold cyan] "
                f"(current: [dim]{info.current}[/dim])\n"
                f"   Run: [bold yellow]uv tool upgrade hud-python[/bold yellow] to update"
            )

            # Display using console info
            console.info(f"[yellow]{update_msg}[/yellow]")
    except Exception:  # noqa: S110
        # Never let version checking disrupt the user's workflow
        pass


def force_version_check() -> VersionInfo | None:
    """Force a version check, bypassing the cache.

    This is useful for explicit version checks or testing.

    Returns:
        VersionInfo if check succeeds, None otherwise.
    """
    # Clear the cache to force a fresh check
    if VERSION_CACHE_FILE.exists():
        with contextlib.suppress(Exception):
            VERSION_CACHE_FILE.unlink()

    return check_for_updates()
