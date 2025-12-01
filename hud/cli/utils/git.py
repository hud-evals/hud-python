"""Git utilities for extracting repository information."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_git_remote_url(cwd: Path | None = None) -> str | None:
    """
    Get the git remote origin URL for the current repository.

    Args:
        cwd: Working directory (defaults to current directory)

    Returns:
        Git remote URL if available, None otherwise
    """
    cwd = cwd or Path.cwd()

    try:
        # Check if we're in a git repository
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            check=True,
        )

        # Get the remote origin URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )

        url = result.stdout.strip()
        if url:
            return normalize_github_url(url)
        return None

    except subprocess.CalledProcessError:
        # Not a git repository or no remote origin
        return None
    except Exception as e:
        logger.debug("Error getting git remote URL: %s", e)
        return None


def normalize_github_url(url: str) -> str:
    """
    Normalize various git URL formats to standard HTTPS GitHub URL.

    Examples:
        git@github.com:user/repo.git -> https://github.com/user/repo
        https://github.com/user/repo.git -> https://github.com/user/repo
        git://github.com/user/repo.git -> https://github.com/user/repo

    Args:
        url: Git remote URL in any format

    Returns:
        Normalized HTTPS GitHub URL
    """
    # Remove trailing .git
    if url.endswith(".git"):
        url = url[:-4]

    # Handle SSH format (git@github.com:user/repo)
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")

    # Handle git:// protocol
    elif url.startswith("git://"):
        url = url.replace("git://", "https://")

    # Ensure HTTPS
    elif not url.startswith("https://") and "github.com:" in url:
        parts = url.split("github.com:")
        url = f"https://github.com/{parts[1]}"

    return url


def get_git_info(cwd: Path | None = None) -> dict[str, Any]:
    """
    Get comprehensive git repository information.

    Args:
        cwd: Working directory (defaults to current directory)

    Returns:
        Dictionary with git info including:
        - remote_url: The remote origin URL
        - branch: Current branch name
        - commit: Current commit hash (short)
    """
    cwd = cwd or Path.cwd()
    info: dict[str, Any] = {}

    # Get remote URL
    info["remote_url"] = get_git_remote_url(cwd)

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        info["branch"] = result.stdout.strip()

        # Get current commit (short hash)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        info["commit"] = result.stdout.strip()

    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        logger.debug("Error getting git info: %s", e)

    return info
