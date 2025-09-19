"""
Remote runner for HUD RL training via API server.

This module implements the new interactive flow for RL training.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

from rich.console import Console

from hud.utils.hud_console import hud_console
from hud.utils.tasks import load_tasks

from . import rl_api
from .config import generate_config_interactive, load_config, save_config
from .presets import get_training_presets

console = Console()

# GPU pricing information
GPU_PRICING = {
    "A100": {"price": "1", "memory": "80GB"},
    "H100": {"price": "2", "memory": "80GB"},
}


def ensure_vllm_deployed(model_name: str, gpu_type: str = "A100", timeout: int = 600) -> None:
    """Deploy vLLM for a model if needed and wait until it's ready.

    Args:
        model_name: The name of the model to deploy vLLM for
        gpu_type: GPU type to use for deployment (e.g., A100, H100)
        timeout: Max seconds to wait for vLLM to be ready
    """
    # Check current model status
    info = rl_api.get_model(model_name)
    if info.vllm_url:
        hud_console.success("vLLM server already running")
        return

    hud_console.info(f"Deploying vLLM server for {model_name}...")
    rl_api.deploy_vllm(model_name, gpu_type=gpu_type)
    hud_console.success("vLLM deployment started")

    hud_console.info("Waiting for vLLM server to be ready...")
    start_time = time.time()
    with hud_console.progress() as progress:
        progress.update("Checking deployment status (see live status on https://app.hud.so/models)")
        while True:
            if time.time() - start_time > timeout:
                hud_console.error("Timeout waiting for vLLM deployment")
                raise ValueError("vLLM deployment timeout")
            info = rl_api.get_model(model_name)
            if info.vllm_url or info.status == "ready":
                hud_console.success(
                    f"vLLM server ready at http://rl.hud.so/v1/models/{model_name}/vllm"
                )
                break
            time.sleep(5)


def run_remote_training(
    tasks_file: str | None,
    model: str | None,
    config_file: Path | None,
    output_dir: str,
) -> None:
    """Run RL training remotely via the API server following the new interactive flow."""
    from hud.settings import settings

    if not settings.api_key:
        hud_console.error("API key not found")
        console.print("[yellow]Please set HUD_API_KEY environment variable[/yellow]")
        raise ValueError("API key not found")

    # Step 1: CONFIRMATION - Load tasks and show example
    if tasks_file:
        tasks = load_tasks(tasks_file)
    else:
        raise ValueError("Tasks file not found")

    # Show example task for confirmation
    hud_console.section_title("Example Task from Dataset")

    if tasks:
        # Display task with truncated values
        task_data = tasks[0].model_dump()
        truncated_data = {}
        max_value_length = 120  # Maximum characters to show per line

        for key, value in task_data.items():
            value_str = str(value)
            if len(value_str) > max_value_length:
                truncated_data[key] = value_str[:max_value_length] + "..."
            else:
                truncated_data[key] = value_str

        hud_console.key_value_table(truncated_data)

        if not hud_console.confirm("Proceed with training on this dataset?", default=True):
            hud_console.error("Training cancelled")
            return

    # Step 2: MODEL SELECTION
    hud_console.section_title("Model Selection")

    # Fetch existing models
    hud_console.info("Fetching your models from https://app.hud.so/models")

    try:
        models = rl_api.list_models()
        # Filter for active/training models and sort by recency
        active_models = [m for m in models if m.status in ["ready", "training"]]
        active_models.sort(key=lambda m: m.created_at or "", reverse=True)

        if active_models or model is None:
            # Build choices
            choices = []
            for m in active_models:
                status_emoji = {
                    "ready": "✅",
                    "training": "🔄",
                    "deploying": "🚀",
                    "pending": "⏳",
                }.get(m.status, "❓")

                choices.append({"name": f"{status_emoji} {m.name} ({m.status})", "value": m.name})

            choices.append({"name": "Create new model", "value": "__new__"})

            if not model:
                if choices:
                    selected = hud_console.select("Select a model:", choices=choices)
                else:
                    selected = "__new__"
                    hud_console.hint("No existing models found. Creating new model...")
            else:
                # Model was provided via CLI
                selected = model

        else:
            selected = "__new__"

        # Handle model selection
        if selected == "__new__":
            # Create new model flow
            hud_console.info("Creating new model...")

            # Ask for model type
            model_type = hud_console.select(
                "Select base model type:",
                choices=[
                    {"name": "Qwen2.5-VL-3B-Instruct", "value": "Qwen/Qwen2.5-VL-3B-Instruct"},
                    # {"name": "Qwen2.5-VL-7B-Instruct", "value": "Qwen/Qwen2.5-VL-7B-Instruct"},
                ],
                default=0,
            )
            from rich.prompt import Prompt

            # Ask for model name
            base_default = model_type.split("/")[-1].lower()
            default_name = base_default
            existing_names = {m.name for m in active_models}
            suffix = 1
            while default_name in existing_names:
                default_name = f"{base_default}-{suffix}"
                suffix += 1

            hud_console.info(f"Enter model name (default: {default_name}):")
            model_name = Prompt.ask("Model name", default=default_name)
            model_name = model_name.replace("/", "-").lower()

            # Create the model with retry on name conflict
            hud_console.info(f"Creating model: {model_name}")
            try:
                rl_api.create_model(model_name, model_type)
                hud_console.success(f"Created model: {model_name}")
                ensure_vllm_deployed(model_name, gpu_type="A100")

            except Exception as e:
                # If the name already exists, suggest a new name and prompt once
                message = str(e)
                if "already exists" in message or "409" in message:
                    alt_name = f"{model_name}-1"
                    i = 1
                    while True:
                        candidate = f"{model_name}-{str(uuid.uuid4())[:4]}"
                        if candidate not in existing_names:
                            alt_name = candidate
                            break
                        i += 1
                    hud_console.warning(
                        f"Model '{model_name}' exists. Suggesting '{alt_name}' instead."
                    )
                    try:
                        from rich.prompt import Prompt as _Prompt

                        chosen = _Prompt.ask("Use different name", default=alt_name)
                        chosen = chosen.replace("/", "-").lower()
                        rl_api.create_model(chosen, model_type)
                        hud_console.success(f"Created model: {chosen}")
                        model_name = chosen
                        ensure_vllm_deployed(model_name, gpu_type="A100")
                    except Exception as e2:
                        hud_console.error(f"Failed to create model: {e2}")
                        raise
                else:
                    hud_console.error(f"Failed to create model: {e}")
                    raise

        else:
            # Existing model selected
            model_name = selected
            model_info = rl_api.get_model(model_name)

            # Check if model is in training
            if model_info.status == "training":
                if hud_console.confirm(
                    f"{model_name} is currently training. Stop current training?", default=False
                ):
                    hud_console.info(f"Stopping training for {model_name}...")
                    try:
                        rl_api.stop_training(model_name)
                        hud_console.success("Training stopped")
                    except Exception as e:
                        hud_console.error(f"Failed to stop training: {e}")
                        raise
                else:
                    hud_console.error("Cannot start new training while model is already training")
                    return

            # Ensure vLLM is deployed
            ensure_vllm_deployed(model_name, gpu_type="A100")
    except KeyboardInterrupt:
        hud_console.dim_info("Training cancelled", "")
        return
    except Exception as e:
        hud_console.error(f"Error during model selection: {e}")
        raise

    # Get final model info
    model_info = rl_api.get_model(model_name)

    # Step 3: TRAINING CONFIG
    hud_console.section_title("Training Configuration")

    if not config_file:
        # Ask about number of GPUs with pricing
        # hud_console.info("GPU Selection (Pricing per GPU):")

        # gpu_table = Table(show_header=True, header_style="bold magenta")
        # gpu_table.add_column("GPU Type", style="cyan")
        # gpu_table.add_column("Memory", style="green")
        # gpu_table.add_column("Price/hr", style="yellow")

        # for gpu, info in GPU_PRICING.items():
        #     gpu_table.add_row(gpu, info["memory"], "see pricing on hud.so")

        # console.print(gpu_table)

        gpu_choice = hud_console.select(
            "Select GPU type:",
            choices=[
                {"name": "A100 80GB", "value": "A100"},
                {"name": "H100 80GB", "value": "H100"},
            ],
            default=0,
        )

        num_gpus = hud_console.select(
            "Number of GPUs:",
            choices=[
                {"name": "1 GPU", "value": 1},
                {"name": "2 GPUs", "value": 2},
                {"name": "4 GPUs", "value": 4},
                {"name": "8 GPUs", "value": 8},
            ],
            default=1,
        )

        # Generate config with presets
        hud_console.info("Generating training configuration...")
        gpu_memory_gb = 80.0 if gpu_choice in ["A100", "H100"] else 48.0
        presets = get_training_presets(gpu_memory_gb)

        config, _ = generate_config_interactive(
            model_name=model_info.base_model,
            presets=presets,
        )

        # Use a short label for tasks (avoid full absolute paths)
        try:
            if tasks_file and Path(tasks_file).exists():
                tasks_label = Path(tasks_file).name
            else:
                # Fallback: last segment of a non-existent path or dataset name
                tasks_label = str(tasks_file).replace("\\", "/").split("/")[-1]
        except Exception:
            tasks_label = str(tasks_file)

        config.job_name = f"RL {model_name} on {tasks_label}"

        # Save config for editing
        temp_config_path = Path(f".rl_config_temp_{model_name}.json")
        save_config(config, temp_config_path)

        # Ask to edit config
        hud_console.info(
            f"Using training configuration from [underline cyan]{temp_config_path.absolute()}[/underline cyan]"  # noqa: E501
        )
        edit_choice = hud_console.select(
            "Would you like to start training?",
            choices=[
                {"name": "🚀 Start training!", "value": "start"},
                {"name": "✏️  Review configuration", "value": "edit"},
                {"name": "❌ Cancel", "value": "cancel"},
            ],
        )

        if edit_choice == "cancel":
            hud_console.error("Training cancelled")
            return
        elif edit_choice == "edit":
            # Open editor
            editor = os.environ.get("EDITOR", "nano")
            hud_console.info(f"Opening {editor} to edit configuration...")

            try:
                subprocess.run([editor, str(temp_config_path)], check=True)  # noqa: S603
                # Reload config
                config = load_config(temp_config_path)
                hud_console.success("Configuration updated")
            except Exception as e:
                hud_console.error(f"Failed to edit config: {e}")
                return

        config_dict = config.to_dict()
    else:
        # Load provided config
        hud_console.info(f"Loading configuration from: {config_file}")
        config = load_config(config_file)
        config_dict = config.to_dict()
        gpu_choice = "A100"  # Default
        num_gpus = 1  # Default for non-interactive mode

    # Launch training
    try:
        rl_api.launch_training(
            model_name=model_name,
            config=config_dict,
            tasks=[task.model_dump() for task in tasks],
            gpu_type=gpu_choice,
            gpu_count=int(num_gpus),
        )

        hud_console.success("Training Started Successfully!")

        hud_console.info(f"See your model {model_name} training on https://app.hud.so/models")
        hud_console.hint("Launch another training run via: hud rl <tasks_file>")
        hud_console.hint("Or evaluate the model via: hud eval <tasks_file>")

    except Exception as e:
        hud_console.error(f"Failed to launch training: {e}")
        raise
