"""Configuration file parsing for hud eval command."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

VALID_CONFIG_KEYS = [
    "agent",
    "source",
    "model",
    "full",
    "max_steps",
    "max_concurrent",
    "group_size",
    "allowed_tools",
    "verbose",
    "very_verbose",
    "vllm_base_url",
]

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
                    config[key] = value

    return config


def generate_default_config(path: str = ".hud_eval_config") -> None:
    """Generate default .hud_eval_config file."""
    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)


def format_settings_for_display(settings: dict[str, Any]) -> str:
    """Format settings dictionary for display in confirmation prompt."""
    lines = ["Evaluation Settings:"]

    display_settings = {k: v for k, v in settings.items() if v is not None}
    if not display_settings:
        return "Evaluation Settings: (none configured)"

    max_key_length = max(len(k) for k in display_settings)

    for key in VALID_CONFIG_KEYS:
        if key in display_settings:
            value = display_settings[key]
            if isinstance(value, Enum):
                value = value.value
            lines.append(f"  {key.ljust(max_key_length)}: {value}")

    return "\n".join(lines)
