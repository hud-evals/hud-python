"""RL training command for HUD CLI."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from hud.cli.utils.tasks import find_tasks_file
from hud.utils.hud_console import hud_console

console = Console()

if TYPE_CHECKING:
    from pathlib import Path


def rl_command(
    tasks_file: str | None = typer.Argument(
        None,
        help="Path to tasks file (JSON/JSONL) or HuggingFace dataset name",
    ),
    model: str | None = typer.Argument(
        None,
        help="Model to train from https://hud.ai/models (default: interactive selection)",
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Path to existing configuration file",
    ),
    output_dir: str = typer.Option(
        "/checkpoints",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    restart: bool = typer.Option(
        False,
        "--restart",
        help="Restart the vLLM server before training",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    # DDP options
    no_ddp: bool = typer.Option(
        False,
        "--no-ddp",
        help="Disable DDP even with multiple GPUs",
    ),
    ddp_gpus: str | None = typer.Option(
        None,
        "--ddp-gpus",
        help="Specific GPUs for DDP (e.g., '0,1,2,3')",
    ),
    vllm_gpu: int | None = typer.Option(
        None,
        "--vllm-gpu",
        help="Specific GPU for vLLM server",
    ),
    # Execution mode options
    local: bool = typer.Option(
        False,
        "--local",
        help="Run training locally instead of using remote API server",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-accept all prompts and use defaults (lazy mode)",
    ),
    vllm_gpu_count: int = typer.Option(
        None,
        "--vllm-gpu-count",
        help="Number of GPUs for vLLM server",
    ),
    skip_vllm_startup: bool = typer.Option(
        False,
        "--skip-vllm-startup",
        help="Skip local vLLM server startup (for internal use)",
    ),
) -> None:
    """Run GRPO reinforcement learning training on tasks."""
    # Configure logging based on verbose flag BEFORE any output
    if not verbose:
        os.environ["HUD_LOG_LEVEL"] = "WARNING"
        logging.basicConfig(level=logging.WARNING, force=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)

        # Suppress INFO logs from various components
        for logger_name in [
            "httpx",
            "hud.agents",
            "hud.utils.design",
            "hud",
            "asyncio",
            "transformers",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger("hud.agents.base").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    hud_console.header("HUD RL Training")

    # Determine execution mode
    use_remote = not local

    if not tasks_file:
        tasks_file = find_tasks_file(tasks_file)
        if not tasks_file:
            hud_console.warning("No tasks file found in current directory")
            hud_console.hint(
                "Download a HF dataset using `hud get <dataset_name>` (e.g., `hud get hud-evals/2048-basic`)"  # noqa: E501
            )
            hud_console.hint("or create a tasks file manually.")
            raise typer.Exit(1)

    # If user ran bare `hud rl`, guide them through remote task conversion flow
    # before proceeding (remote only)
    if use_remote:
        try:
            from hud.cli.flows.tasks import convert_tasks_to_remote

            console.print("[cyan]Preparing remote training tasks...[/cyan]")
            tasks_file = convert_tasks_to_remote(tasks_file)
        except typer.Exit:
            raise
        except Exception as e:
            hud_console.warning(f"[red]Tasks file is not valid for remote training: {e!s}[/red]")
            hud_console.hint("Either ensure the tasks file has remote urls")
            hud_console.hint("Or rerun `hud rl` within an environment directory")
            raise typer.Exit(1) from e

        try:
            from .remote_runner import run_remote_training

            run_remote_training(
                tasks_file=tasks_file,
                model=model,
                config_file=config_file,
                output_dir=output_dir,
                vllm_gpu_count=vllm_gpu_count,
                yes=yes,
            )
            return
        except Exception as e:
            console.print(f"[red]❌ Remote training failed: {e!s}[/red]")
            raise typer.Exit(1) from e

    # Local execution flow delegated to local_runner (imports heavy deps lazily)
    from .local_runner import run_local_training

    run_local_training(
        tasks_file=tasks_file,
        model=model,
        config_file=config_file,
        output_dir=output_dir,
        yes=yes,
        restart=restart,
        verbose=verbose,
        no_ddp=no_ddp,
        ddp_gpus=ddp_gpus,
        vllm_gpu=vllm_gpu,
        skip_vllm_startup=skip_vllm_startup,
    )


# Export the command function
__all__ = ["rl_command"]
