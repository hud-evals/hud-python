from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from hud.rl.config import Config

console = Console()


def display_config_summary(
    config: Config,
    *,
    tasks_count: int | None = None,
    trainer_gpus: int | None = None,
) -> None:
    """Render a concise summary of the key configuration values."""

    table = Table(title="📋 Training Configuration", title_style="bold cyan")
    table.add_column("Setting", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Base model", config.base_model)
    table.add_row("Training steps", str(config.training_steps))
    table.add_row("Batch size", str(config.batch_size))
    table.add_row("Mini-batch size", str(config.mini_batch_size))
    table.add_row("Group size", str(config.group_size))
    trainer_value = trainer_gpus if trainer_gpus is not None else config.num_gpus
    table.add_row("Trainer GPUs", str(trainer_value))
    table.add_row("Actor max steps", str(config.actor.max_steps_per_episode))
    table.add_row("Actor max tokens", str(config.actor.max_new_tokens))
    table.add_row("Actor temperature", f"{config.actor.temperature:.2f}")
    table.add_row(
        "Max parallel episodes",
        str(config.actor.max_parallel_episodes),
    )
    table.add_row("Learning rate", f"{config.training.optimizer.lr:.2e}")

    if tasks_count is not None:
        table.add_row("Tasks count", str(tasks_count))

    console.print(table)
