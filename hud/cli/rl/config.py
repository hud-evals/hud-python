"""Configuration generation and management for RL training."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.console import Console

from hud.rl.config import Config, validate_vl_model
from hud.utils.hud_console import hud_console

from .display import display_preset_table
from .presets import estimate_memory_usage

if TYPE_CHECKING:
    from pathlib import Path
console = Console()


def generate_config_interactive(
    model_name: str,
    presets: list[dict[str, Any]],
    yes: bool = False,
) -> tuple[Config, float]:
    """Generate RL training configuration interactively."""
    # Validate model is a VL model
    validate_vl_model(model_name)

    # Display preset options
    if not yes:
        display_preset_table(presets, 80.0)  # Assuming A100 80GB

    # Let user select preset
    if yes:
        # Use default preset (Balanced if available, otherwise first)
        preset_choice = 1 if len(presets) > 1 else 0
        selected_preset = presets[preset_choice]
        hud_console.info(f"Auto-selecting preset: {selected_preset['name']} (--yes mode)")
    else:
        preset_choice = hud_console.select(
            "Select a training configuration preset:",
            choices=[{"name": p["name"], "value": i} for i, p in enumerate(presets)],
            default=1 if len(presets) > 1 else 0,  # Default to "Balanced" if available
        )
        selected_preset = presets[preset_choice]  # type: ignore

    # Use preset values directly
    max_steps_per_episode = selected_preset["max_steps_per_episode"]

    # Calculate memory estimate
    max_pixels = 256 * 28 * 28
    estimated_memory = estimate_memory_usage(
        selected_preset["mini_batch_size"],
        max_steps_per_episode,
        selected_preset["max_new_tokens"],
        max_pixels,
    )

    config_adds = {
        "actor": {
            "max_new_tokens": selected_preset["max_new_tokens"],
            "max_parallel_episodes": selected_preset["batch_size"],
            "max_steps_per_episode": selected_preset["max_steps_per_episode"],
            "force_tool_choice": True,
        },
        "training": {
            "mini_batch_size": selected_preset["mini_batch_size"],
            "group_size": selected_preset["group_size"],
            "batch_size": selected_preset["batch_size"],
            "lr": selected_preset["lr"],
            "epochs": selected_preset["epochs"],
        },
        "verbose": True,
    }

    # Create config
    config = Config.from_dict(config_adds)

    return config, estimated_memory


def save_config(config: Config, path: Path) -> None:
    """Save configuration to a JSON file."""
    config_dict = config.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
        f.write("\n")  # Add newline at end of file

    if not path.name.startswith("."):  # Don't show message for temp files
        console.print(f"[green]✅ Configuration saved to {path}[/green]")


def load_config(path: Path) -> Config:
    """Load configuration from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Use Config.from_dict which handles missing fields gracefully
    return Config.from_dict(data)
