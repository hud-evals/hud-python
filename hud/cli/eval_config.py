"""Configuration file parsing for hud eval command."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from rich import box
from rich.table import Table

from hud.utils import hud_console

VALID_CONFIG_KEYS = [
    "agent",
    "source",
    "task_id",
    "model",
    "full",
    "max_steps",
    "max_concurrent",
    "group_size",
    "allowed_tools",
    "verbose",
    "very_verbose",
    "vllm_base_url",
    "remote",
    "batch_size",
]

CONFIG_TYPES = {
    "max_concurrent": int,
    "group_size": int,
    "max_steps": int,
    "full": bool,
    "verbose": bool,
    "very_verbose": bool,
    "remote": bool,
    "batch_size": int,
}

DEFAULT_CONFIG_TEMPLATE = """# HUD Eval Configuration
# This file configures default settings for the 'hud eval' command
# Command-line arguments override these settings
# Uncomment and modify the settings you want to change

# Agent backend to use (claude, openai, vllm, gemini, or litellm)
# agent=claude

# Model name for the chosen agent
# model=

# Run the entire dataset (omit for single-task debug mode)
# full=false

# Maximum concurrent tasks (1-200 recommended, prevents rate limits, default: 30)
# max_concurrent=30

# Maximum steps per task (default: 10 for single, 50 for full)
# max_steps=

# Comma-separated list of allowed tools
# allowed_tools=

# Enable verbose output from the agent
# verbose=false

# Enable debug-level logs for maximum visibility
# very_verbose=false

# Base URL for vLLM server (when using --agent vllm)
# vllm_base_url=

# Number of times to run each task (similar to RL training, default: 1)
# group_size=1

# Run evaluation remotely on HUD infrastructure (default: false)
# remote=false

# Batch size for remote API submissions (default: 10)
# batch_size=10
"""


def parse_value(value: str) -> Any:
    """Convert string value to appropriate type."""
    value = value.strip()
    if not value:
        return None
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.isdigit():
        return int(value)
    return value


def load_eval_config(path: str = ".hud_eval_config") -> dict[str, Any]:
    """Load and parse .hud_eval_config file."""
    config_path = Path(path)
    if not config_path.exists():
        generate_default_config()
        hud_console.info("Generated .hud_eval_config")
        return {}

    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in VALID_CONFIG_KEYS:
                    continue
                value = parse_value(value)
                if value is not None:
                    if key in CONFIG_TYPES:
                        expected_type = CONFIG_TYPES[key]
                        if not isinstance(value, expected_type):
                            hud_console.warning(
                                f"Config error: '{key}' expects {expected_type.__name__}, "
                                f"got '{value}'. Using default."
                            )
                            continue
                    config[key] = value

    return config


def generate_default_config(path: str = ".hud_eval_config") -> None:
    """Generate default .hud_eval_config file."""
    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)


def display_eval_settings(settings: dict[str, Any]) -> None:
    """Display settings in a formatted table."""
    display_settings = {k: v for k, v in settings.items() if v is not None}

    if not display_settings:
        hud_console.info("Evaluation Settings: (none configured)")
        return

    table = Table(
        title="Evaluation Settings",
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="green")

    # User requested order
    order = [
        "agent",
        "source",
        "task_id",
        "full",
        "remote",
        "batch_size",
        "max_steps",
        "max_concurrent",
        "group_size",
        "verbose",
        "very_verbose",
    ]
    # Add remaining keys
    for key in VALID_CONFIG_KEYS:
        if key not in order:
            order.append(key)

    seen = set()
    for key in order:
        if key in display_settings:
            value = display_settings[key]
            if isinstance(value, Enum):
                value = value.value
            table.add_row(key, str(value))
            seen.add(key)

    # Add any keys in settings but not in VALID_CONFIG_KEYS (shouldn't happen but good for safety)
    for key, value in display_settings.items():
        if key not in seen:
            if isinstance(value, Enum):
                value = value.value
            table.add_row(key, str(value))

    hud_console.console.print(table)
