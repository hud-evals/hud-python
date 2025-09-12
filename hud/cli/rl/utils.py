"""Shared utilities for RL commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()

logger = logging.getLogger(__name__)


def read_lock_file() -> dict[str, Any]:
    """Read and parse hud.lock.yaml file."""
    lock_file = Path("hud.lock.yaml")
    if not lock_file.exists():
        return {}

    try:
        with open(lock_file) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        hud_console.warning(f"Could not read hud.lock.yaml: {e}")
        return {}


def write_lock_file(data: dict[str, Any]) -> bool:
    """Write data to hud.lock.yaml file."""
    lock_file = Path("hud.lock.yaml")

    try:
        with open(lock_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        hud_console.warning(f"Could not write hud.lock.yaml: {e}")
        return False


def get_mcp_config_from_lock() -> dict[str, Any] | None:
    """Get MCP configuration from lock file."""
    lock_data = read_lock_file()

    # Check if there's an image reference
    image = lock_data.get("image")
    if image:
        return {
            "hud": {
                "url": "https://mcp.hud.so/v3/mcp",
                "headers": {"Authorization": "Bearer $HUD_API_KEY", "Mcp-Image": image},
            }
        }

    return None


def get_primary_dataset() -> str | None:
    """Get primary dataset name from lock file."""
    lock_data = read_lock_file()
    return lock_data.get("primary_dataset", {}).get("name")


def get_image_from_lock() -> str | None:
    """Get image name from lock file."""
    lock_data = read_lock_file()
    return lock_data.get("image")


def detect_image_name() -> str | None:
    """Try to detect image name from various sources."""
    # First check lock file
    image = get_image_from_lock()
    if image:
        return image

    # Check pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        try:
            import tomllib

            with open(pyproject, "rb") as f:
                data = tomllib.load(f)

            # Check for hud.image_name
            image = data.get("tool", {}).get("hud", {}).get("image_name")
            if image:
                return image

            # Use project name
            name = data.get("project", {}).get("name")
            if name:
                return f"{name}:latest"
        except Exception:
            logger.warning("Failed to load pyproject.toml")

    # Use directory name as last resort
    return f"{Path.cwd().name}:latest"


def validate_dataset_name(name: str) -> bool:
    """Validate HuggingFace dataset name format."""
    if not name:
        return False

    if "/" not in name:
        hud_console.error(f"Invalid dataset name: {name}")
        hud_console.info("Dataset name should be in format: org/dataset")
        return False

    parts = name.split("/")
    if len(parts) != 2:
        hud_console.error(f"Invalid dataset name: {name}")
        return False

    org, dataset = parts
    if not org or not dataset:
        hud_console.error(f"Invalid dataset name: {name}")
        return False

    # Check for valid characters (alphanumeric, dash, underscore)
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", org) or not re.match(r"^[a-zA-Z0-9_-]+$", dataset):
        hud_console.error(f"Invalid characters in dataset name: {name}")
        hud_console.info("Use only letters, numbers, dashes, and underscores")
        return False

    return True


def create_tasks_template() -> list[dict[str, Any]]:
    """Create a template for tasks.json file."""
    return [
        {
            "id": "example-task-001",
            "prompt": "Complete the first TODO item in the list",
            "mcp_config": {
                "# TODO": "Add your MCP configuration here",
                "# Example for remote": {
                    "hud": {
                        "url": "https://mcp.hud.so/v3/mcp",
                        "headers": {
                            "Authorization": "Bearer $HUD_API_KEY",
                            "Mcp-Image": "your-org/your-env:latest",
                        },
                    }
                },
                "# Example for local": {
                    "local": {"command": "docker", "args": ["run", "--rm", "-i", "your-env:latest"]}
                },
            },
            "setup_tool": {"name": "setup", "arguments": {"name": "todo_seed", "num_items": 5}},
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "todo_completed", "expected_count": 1},
            },
            "metadata": {"difficulty": "easy", "category": "task_completion"},
        }
    ]
