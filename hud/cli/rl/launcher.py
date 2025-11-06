from __future__ import annotations

import contextlib
import time
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.prompt import Prompt

from hud.cli.rl.celebrate import show_confetti_async
from hud.cli.rl.display import display_config_summary
from hud.cli.rl.viewer import show_json_interactive
from hud.cli.rl.wait_utils import wait_for_enter_cancel_or_change
from hud.rl.config import Config
from hud.utils.hud_console import hud_console
from hud.utils.tasks import load_tasks
from hud.types import Task

from . import rl_api

console = Console()


def ensure_vllm_deployed(
    model_name: str, config: dict[str, Any], gpu_type: str = "A100", gpu_count: int = 1, timeout: int = 600
) -> None:
    """Deploy vLLM for a model if needed and wait until it's ready."""

    info = rl_api.get_model(model_name)
    if info.vllm_url:
        hud_console.success("vLLM server already running")
        return

    hud_console.info(f"Deploying vLLM server for {model_name}...")
    rl_api.deploy_vllm(model_name, config=config, gpu_type=gpu_type, gpu_count=gpu_count)
    hud_console.success("vLLM deployment started")

    hud_console.info("Waiting for vLLM server to be ready...")
    start_time = time.time()
    with hud_console.progress() as progress:
        while True:
            progress.update(
                f"Checking deployment status. See https://hud.ai/models"
            )
            if time.time() - start_time > timeout:
                hud_console.error("Timeout waiting for vLLM deployment")
                raise ValueError("vLLM deployment timeout")
            info = rl_api.get_model(model_name)
            if info.status == "ready":
                hud_console.success(
                    f"vLLM server ready at http://rl.hud.ai/v1/models/{model_name}/vllm"
                )
                break
            time.sleep(0.5)


def launch_training(
    tasks_file: str | None,
    model: str | None,
    config_file: Path | None,
    vllm_gpu_count: int | None = None,
    yes: bool = False,
) -> None:
    from hud.settings import settings

    if not settings.api_key:
        hud_console.error("API key not found")
        console.print(
            "[yellow]Set it in your environment or run: hud set HUD_API_KEY=your-key-here[/yellow]"
        )
        raise ValueError("API key not found")

    hud_console.header("HUD RL Training")

    tasks = _load_and_preview_tasks(tasks_file, yes=yes)

    try:
        model_name, model_info, _ = _select_model(
            desired_model=model,
            config_file=config_file,
            yes=yes,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        hud_console.error(f"Error during model selection: {exc}")
        raise

    gpu_choice, num_gpus, temp_path, config_obj = _prepare_training_config(
        model_info=model_info,
        tasks_file=tasks_file,
        config_file=config_file,
        tasks_count=len(tasks),
        yes=yes,
    )

    vllm_count = vllm_gpu_count if vllm_gpu_count and vllm_gpu_count > 0 else 1
    try:
        ensure_vllm_deployed(
            model_name,
            config=config_obj.vllm.model_dump(),
            gpu_type="A100",
            gpu_count=vllm_count
        )
    except Exception as exc:
        hud_console.error(f"Failed to prepare vLLM server: {exc}")
        raise

    try:
        try:
            show_confetti_async(console)
        except Exception:
            hud_console.info("Launching training...")

        rl_api.launch_training(
            model_name=model_name,
            config=config_obj.model_dump(),
            tasks=[task.model_dump() for task in tasks],
            gpu_type=gpu_choice,
            gpu_count=num_gpus,
        )

        hud_console.info(f"Your model {model_name} has started training")
        hud_console.hint("Launch another training run via: hud rl <tasks_file>")
        hud_console.hint("Or evaluate the model via: hud eval <tasks_file>")
    except Exception as exc:
        hud_console.error(f"Failed to launch training: {exc}")
        raise
    finally:
        if temp_path:
            with contextlib.suppress(Exception):
                temp_path.unlink()


def _load_and_preview_tasks(tasks_file: str | None, *, yes: bool) -> list[Task]:
    if not tasks_file:
        raise ValueError("Tasks file not found")

    tasks: list[Task] = load_tasks(tasks_file)  # type: ignore[arg-type]

    if tasks and not yes:
        try:
            show_json_interactive(tasks[0].model_dump(), title="Task Preview")
        except Exception as exc:
            hud_console.warning(f"Interactive viewer failed: {exc}")
    return tasks


def _select_model(
    *, desired_model: str | None, config_file: Path | None, yes: bool
) -> tuple[str, rl_api.RLModelInfo, bool]:
    models = rl_api.list_models()
    active_models = [m for m in models if m.status in {"ready", "training"}]
    active_models.sort(key=lambda m: m.created_at or "", reverse=True)
    existing_names = {m.name for m in active_models}

    if desired_model:
        selected = desired_model
    elif active_models:
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

        if yes:
            selected = "__new__"
            hud_console.info("Auto-creating new model (--yes mode)")
        else:
            selected = hud_console.select("Select a model:", choices=choices)
    else:
        selected = "__new__"
        hud_console.hint("No existing models found. Creating new model...")

    if selected == "__new__":
        base_model = _choose_base_model(config_file, yes=yes)
        model_name = _choose_model_name(base_model, existing_names, yes=yes)
        _create_model(model_name, base_model, existing_names, yes=yes)
        model_info = rl_api.get_model(model_name)
        return model_name, model_info, True

    model_name = selected
    model_info = rl_api.get_model(model_name)

    if model_info.status == "training":
        if yes:
            hud_console.warning(
                f"{model_name} is already training, skipping (--yes mode)", stderr=True
            )
            raise typer.Exit(0)
        if hud_console.confirm(
            f"{model_name} is currently training. Stop current training?", default=False
        ):
            hud_console.info(f"Stopping training for {model_name}...")
            rl_api.stop_training(model_name)
            hud_console.success("Training stopped")
        else:
            hud_console.error("Cannot start new training while model is already training")
            raise typer.Exit(0)

    return model_name, model_info, False


def _choose_base_model(config_file: Path | None, *, yes: bool) -> str:
    if config_file:
        with contextlib.suppress(Exception):
            cfg = Config.from_file(config_file)
            base = getattr(cfg, "base_model", None)
            if base:
                return str(base)

    if yes:
        hud_console.info("Auto-selecting base model: Qwen/Qwen2.5-VL-3B-Instruct (--yes mode)")
        return "Qwen/Qwen2.5-VL-3B-Instruct"

    return hud_console.select(
        "Select base model type:",
        choices=[
            {"name": "Qwen2.5-VL-3B-Instruct", "value": "Qwen/Qwen2.5-VL-3B-Instruct"},
            {"name": "Qwen2.5-3B-Instruct", "value": "Qwen/Qwen2.5-3B-Instruct"},
        ],
        default=0,
    )


def _choose_model_name(base_model: str, existing: set[str], *, yes: bool) -> str:
    base_default = base_model.split("/")[-1].lower()
    candidate = base_default
    suffix = 1
    while candidate in existing:
        candidate = f"{base_default}-{suffix}"
        suffix += 1

    if yes:
        hud_console.info(f"Auto-using model name: {candidate} (--yes mode)")
        return candidate

    hud_console.info(f"Enter model name (default: {candidate}):")
    return Prompt.ask("Model name", default=candidate).replace("/", "-").lower()


def _create_model(name: str, base_model: str, existing: set[str], *, yes: bool) -> None:
    try:
        rl_api.create_model(name, base_model)
        hud_console.success(f"Created model: {name}")
        return
    except Exception as exc:
        message = str(exc)
        if "already exists" not in message and "409" not in message:
            raise

    while True:
        alt_name = f"{name}-{str(uuid.uuid4())[:4]}"
        if alt_name not in existing:
            break

    if yes:
        chosen = alt_name
        hud_console.info(f"Auto-using suggested name: {chosen} (--yes mode)")
    else:
        chosen = Prompt.ask("Name taken. Use different name", default=alt_name)
    chosen = chosen.replace("/", "-").lower()
    rl_api.create_model(chosen, base_model)
    hud_console.success(f"Created model: {chosen}")


def _prepare_training_config(
    *,
    model_info: rl_api.RLModelInfo,
    tasks_file: str | None,
    config_file: Path | None,
    tasks_count: int,
    yes: bool,
) -> tuple[str, int, Path | None, Config]:
    gpu_choice, num_gpus = _choose_gpu_config(None, yes=yes)

    if config_file:
        hud_console.info(f"Loading configuration from: {config_file}")
        config = Config.from_file(config_file)
        if getattr(config, "num_gpus", None) is None:
            config.num_gpus = num_gpus
        _display_summary(config, tasks_count, gpu_choice, num_gpus)
        return gpu_choice, num_gpus, None, config

    hud_console.info("Generating training configuration...")
    config = _create_default_config(
        model_name=model_info.base_model,
        gpu_type=gpu_choice,
        num_gpus=num_gpus,
    )

    try:
        if tasks_file and Path(tasks_file).exists():
            tasks_label = Path(tasks_file).name
        else:
            tasks_label = str(tasks_file).replace("\\", "/").split("/")[-1]
    except Exception:
        tasks_label = str(tasks_file)
    config.job_name = f"RL {tasks_label} | {model_info.name}"

    temp_config_path = Path(f".rl_config_temp_{model_info.name}.json")
    config.save_to_file(temp_config_path)

    _display_summary(config, tasks_count, gpu_choice, num_gpus)

    hud_console.info(
        f"Edit configuration at: [underline]{temp_config_path}[/underline]"
    )
    config = _review_config_file(
        temp_config_path,
        config,
        gpu_choice,
        num_gpus,
        tasks_count,
        yes=yes,
        summary_already_shown=True,
    )
    return gpu_choice, num_gpus, temp_config_path, config


def _choose_gpu_config(preferred_count: int | None, *, yes: bool) -> tuple[str, int]:
    if yes:
        gpu_choice = "A100"
    else:
        gpu_choice = hud_console.select(
            "Select GPU type:",
            choices=[
                {"name": "A100 80GB", "value": "A100"},
                {"name": "H100 80GB", "value": "H100"},
            ],
            default=0,
        )

    if preferred_count is not None and preferred_count > 0:
        num_gpus = preferred_count
    elif yes:
        num_gpus = 2
    else:
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
    return gpu_choice, int(max(1, num_gpus))


def _create_default_config(*, model_name: str, gpu_type: str, num_gpus: int) -> Config:
    cfg = Config()
    cfg.base_model = model_name
    cfg.num_gpus = max(1, int(num_gpus))

    normalized_gpu = gpu_type.upper()
    if cfg.num_gpus <= 2:
        cfg.group_size = 6
        cfg.batch_size = 12
        cfg.mini_batch_size = 1
    else:
        cfg.group_size = 8
        cfg.batch_size = 16
        cfg.mini_batch_size = 2

    cfg.actor.max_parallel_episodes = cfg.batch_size
    cfg.actor.max_steps_per_episode = 5 if normalized_gpu.startswith("A") else 6
    cfg.actor.max_new_tokens = 1024 if normalized_gpu.startswith("A") else 1536

    cfg.training.optimizer.lr = 3e-5
    cfg.verbose = True
    return cfg


def _display_summary(config: Config, tasks_count: int, gpu_choice: str, num_gpus: int) -> None:
    display_config_summary(
        config,
        tasks_count=tasks_count,
        trainer_gpus=num_gpus,
    )


def _review_config_file(
    path: Path,
    config: Config,
    gpu_choice: str,
    num_gpus: int,
    tasks_count: int,
    *,
    yes: bool,
    summary_already_shown: bool = False,
) -> Config:
    if yes:
        if not summary_already_shown:
            _display_summary(config, tasks_count, gpu_choice, num_gpus)
        return config

    summary_pending = not summary_already_shown

    while True:
        if summary_pending:
            _display_summary(config, tasks_count, gpu_choice, num_gpus)
            summary_pending = False
        console.print(
            "\n[dim]Edit the config file above if needed, then save.[/dim]\n"
            "[bold]Press Enter to start training[/bold], or press 'q' to cancel."
        )

        start_training, cancelled, changed = wait_for_enter_cancel_or_change(path)
        if cancelled:
            hud_console.error("Training cancelled")
            raise typer.Exit(1)
        if start_training:
            break
        if changed:
            hud_console.info("Detected configuration changes. Reloading summary...")
            config = Config.from_file(path)
            summary_pending = True
    return config
