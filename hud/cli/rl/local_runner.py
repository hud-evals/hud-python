"""
Local runner for HUD RL training.

This module encapsulates the local training flow and imports heavy
dependencies (torch, transformers, etc.) only when actually running
locally. The CLI entrypoint should import this module lazily to avoid
pulling heavy deps during remote-only usage.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from hud.rl.config import validate_vl_model
from hud.utils.hud_console import hud_console
from hud.utils.tasks import load_tasks

console = Console()


def run_local_training(
    *,
    tasks_file: str,
    model: str | None,
    config_file: Path | None,
    output_dir: str,
    yes: bool,
    restart: bool,
    verbose: bool,
    no_ddp: bool,
    ddp_gpus: str | None,
    vllm_gpu: int | None,
    skip_vllm_startup: bool,
) -> None:
    """Run RL training locally on the current machine.

    Heavy modules are imported inside this function to avoid import-time side effects
    during remote-only runs.
    """
    # Light-weight utilities
    from .config import generate_config_interactive, load_config, save_config
    from .display import display_config_summary, display_gpu_info
    from .gpu import detect_cuda_devices, validate_gpu_memory
    from .presets import get_training_presets

    # Python version compatibility warning for vLLM
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 13:
        console.print("[red]⚠️  Warning: Python 3.13+ detected![/red]")
        console.print("[yellow]vLLM has compatibility issues with Python 3.13.[/yellow]")
        console.print("[yellow]Recommended: Use Python 3.12 or 3.11[/yellow]")
        console.print("\n[dim]To create a new environment with Python 3.12:[/dim]")
        console.print("[dim]  1. Exit this shell: exit[/dim]")
        console.print("[dim]  2. Remove current venv: sudo rm -rf .venv[/dim]")
        console.print("[dim]  3. Create new venv: uv venv --python 3.12[/dim]")
        console.print("[dim]  4. Install dependencies: uv pip install -e '.[rl]'[/dim]")

        try:
            import typer

            if not yes:
                if not typer.confirm("\nDo you want to continue anyway?", default=False):
                    raise typer.Exit(1)
            else:
                hud_console.warning("Auto-continuing despite Python 3.13+ (--yes mode)")
        except Exception as e:
            hud_console.warning(f"Failed to confirm: {e}")
            return

    # Step 1: Validate CUDA devices
    console.print("[yellow]Checking GPU availability...[/yellow]")
    gpu_info = detect_cuda_devices()

    if not gpu_info["available"]:
        console.print(f"[red]❌ {gpu_info['error']}[/red]")
        console.print("[yellow]RL training requires CUDA-capable GPUs[/yellow]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    display_gpu_info(gpu_info)

    # Perform GPU health check (imports torch lazily)
    all_gpu_indices = [device["index"] for device in gpu_info["devices"]]
    from .gpu_utils import health_check_gpus  # heavy import (torch)

    health_results = health_check_gpus(all_gpu_indices)

    if not health_results["all_healthy"]:
        console.print("\n[yellow]⚠️  Some GPUs failed health checks![/yellow]")
        console.print(
            f"[yellow]Unhealthy GPUs: {list(health_results['unhealthy_gpus'].keys())}[/yellow]"
        )

        if not health_results["healthy_gpus"]:
            console.print("[red]❌ No healthy GPUs available for training![/red]")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return

        console.print(
            f"\n[cyan]You have {len(health_results['healthy_gpus'])} healthy GPUs available.[/cyan]"
        )

        try:
            import typer

            if yes:
                continue_training = True
                hud_console.info("Auto-continuing with healthy GPUs only (--yes mode)")
            else:
                continue_training = typer.confirm(
                    "\nContinue with healthy GPUs only?", default=True
                )
        except Exception:
            continue_training = True

        if not continue_training:
            healthy_str = ",".join(map(str, health_results["healthy_gpus"]))
            console.print("\n[yellow]Exiting. Please resolve GPU issues and try again.[/yellow]")
            console.print("\n[cyan]💡 Tip: To use only healthy GPUs, you can run:[/cyan]")
            console.print(f"[white]hud rl {tasks_file} --ddp-gpus {healthy_str} --local[/white]\n")
            try:
                import typer

                raise typer.Exit(0)
            except Exception:
                return
        else:
            # Continue with healthy GPUs only
            gpu_info["devices"] = [
                d for d in gpu_info["devices"] if d["index"] in health_results["healthy_gpus"]
            ]
            console.print(
                f"\n[green]✅ Continuing with {len(gpu_info['devices'])} healthy GPUs[/green]"
            )

    # Get primary GPU memory for configuration
    primary_gpu = gpu_info["devices"][0]
    gpu_memory_gb = primary_gpu["memory_gb"]

    # Validate GPU memory for 3B model
    if not validate_gpu_memory(gpu_memory_gb, "3B"):
        console.print(f"[red]❌ Insufficient GPU memory ({gpu_memory_gb:.1f} GB)[/red]")
        console.print("[yellow]Qwen 2.5 VL 3B requires at least 12 GB of GPU memory[/yellow]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    # Step 2: Load and validate tasks
    if tasks_file:
        console.print(f"\n[cyan]Loading tasks from: {tasks_file}[/cyan]")
    else:
        possible_files = ["tasks.json", "tasks.jsonl", "browser_2048_tasks.jsonl"]
        for f in possible_files:
            if Path(f).exists():
                tasks_file = f
                console.print(f"[green]Auto-detected tasks file: {f}[/green]")
                break

        if not tasks_file:
            console.print("[red]❌ No tasks file specified or auto-detected[/red]")
            console.print(
                "[yellow]Please provide a tasks file or create one of: tasks.json, tasks.jsonl[/yellow]"  # noqa: E501
            )
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return

    tasks = load_tasks(tasks_file)
    console.print(f"[green]✅ Loaded {len(tasks)} tasks[/green]")

    invalid_tasks: list[str] = []
    for i, task in enumerate(tasks):
        if not hasattr(task, "prompt") or not task.prompt:  # type: ignore
            invalid_tasks.append(f"Task {i}: missing 'prompt' field")
        if not hasattr(task, "mcp_config") or not task.mcp_config:  # type: ignore
            invalid_tasks.append(f"Task {i}: missing 'mcp_config' field")

    if invalid_tasks:
        console.print("[red]❌ Invalid tasks found:[/red]")
        for error in invalid_tasks[:5]:
            console.print(f"  - {error}")
        if len(invalid_tasks) > 5:
            console.print(f"  ... and {len(invalid_tasks) - 5} more")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    # Step 3: Model selection (if not provided)
    if model is None and not config_file:
        if yes:
            model = "Qwen/Qwen2.5-VL-3B-Instruct"  # Default model in yes mode
            hud_console.info(f"Auto-selecting model: {model} (--yes mode)")
        else:
            model = hud_console.select(
                "Select a model for RL training:",
                choices=[
                    {
                        "name": "Qwen 2.5 VL 3B (Recommended - Vision-Language)",
                        "value": "Qwen/Qwen2.5-VL-3B-Instruct",
                    },
                    {"name": "Custom model", "value": "custom"},
                ],
                default=0,
            )

            if model == "custom":
                console.print("Enter the model name (HuggingFace ID):")
                model = input().strip()

    # try to get model from config file
    if config_file:
        console.print(f"\n[cyan]Loading configuration from: {config_file}[/cyan]")
        config = load_config(config_file)
        if hasattr(config, "model") and hasattr(config.model, "base_model"):
            if model is None:
                model = config.model.base_model
            else:
                console.print(
                    f"[yellow]Model already set to {model}, using that instead "
                    f"of {config.model.base_model}[/yellow] (override)"
                )

    if model is None:
        console.print("[red]❌ No model specified either through CLI or config file[/red]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    # Validate model is a VL model (whether provided via CLI or selected)
    try:
        validate_vl_model(model)
    except ValueError as e:
        console.print(f"\n[red]❌ {e}[/red]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    # Step 4: Generate or load configuration
    if config_file:
        console.print(f"\n[cyan]Loading configuration from: {config_file}[/cyan]")
        config = load_config(config_file)

        # Validate model from config
        if hasattr(config, "model") and hasattr(config.model, "base_model"):
            try:
                validate_vl_model(config.model.base_model)
            except ValueError as e:
                console.print(f"\n[red]❌ {e}[/red]")
                try:
                    import typer

                    raise typer.Exit(1)
                except Exception:
                    return

        # Estimate memory for display
        from .presets import estimate_memory_usage

        estimated_memory = estimate_memory_usage(
            config.training.mini_batch_size,
            config.actor.max_steps_per_episode,
            config.actor.max_new_tokens,
            config.model.max_pixels,
        )
    else:
        console.print("\n[cyan]Generating training configuration...[/cyan]")
        # Get number of GPUs for preset scaling
        num_training_gpus = 1  # Default, will be adjusted later
        if len(gpu_info["devices"]) > 2:
            num_training_gpus = len(gpu_info["devices"]) - 1  # Reserve 1 for vLLM
            console.print(
                f"[yellow]Note: Episodes will be scaled for {num_training_gpus} training GPUs[/yellow]\n"  # noqa: E501
            )

        presets = get_training_presets(gpu_memory_gb)
        config, estimated_memory = generate_config_interactive(
            model_name=model,
            presets=presets,
            yes=yes,
        )

    # Step 5: Save temporary config and display summary
    temp_config_path = Path(".rl_config_temp.json")
    save_config(config, temp_config_path)
    console.print(f"\n[cyan]📝 Configuration saved to: {temp_config_path.absolute()}[/cyan]")
    console.print("[yellow]You can edit this file before starting training.[/yellow]")

    # Display configuration summary
    display_config_summary(config, len(tasks), gpu_info, estimated_memory)

    # Step 6: Ask for confirmation (skip if config was provided or in yes mode)
    if not config_file and not yes:
        console.print("\n[bold yellow]Options:[/bold yellow]")
        console.print("  • Type [green]'start'[/green] to begin training")
        console.print("  • Type [cyan]'edit'[/cyan] to open config in your editor")
        console.print("  • Type [red]'cancel'[/red] to abort")
        console.print("\n[bold]Your choice:[/bold] ", end="")

        while True:
            choice = input().strip().lower()

            if choice == "start":
                config = load_config(temp_config_path)  # Reload config in case it was edited
                break
            elif choice == "edit":
                editor = os.environ.get("EDITOR", "nano")

                if editor == "nano":
                    console.print("\n[cyan]Opening config in nano editor...[/cyan]")
                    console.print("[yellow]Tips:[/yellow]")
                    console.print("  • Edit the configuration values as needed")
                    console.print("  • Press [bold]Ctrl+O[/bold] then [bold]Enter[/bold] to save")
                    console.print("  • Press [bold]Ctrl+X[/bold] to exit")
                    console.print("  • Press [bold]Ctrl+C[/bold] to cancel without saving\n")
                    input("Press Enter to continue...")

                try:
                    subprocess.run([editor, str(temp_config_path)], check=True)  # noqa: S603
                    # Reload and display updated config
                    config = load_config(temp_config_path)
                    from .presets import estimate_memory_usage as _estimate_memory

                    estimated_memory = _estimate_memory(
                        config.training.mini_batch_size,
                        config.actor.max_steps_per_episode,
                        config.actor.max_new_tokens,
                        config.model.max_pixels,
                    )
                    display_config_summary(config, len(tasks), gpu_info, estimated_memory)
                    console.print(
                        "\n[bold]Type 'start' to begin or 'cancel' to abort:[/bold] ", end=""
                    )
                except subprocess.CalledProcessError:
                    console.print(
                        "\n[yellow]Editor closed without saving or was cancelled.[/yellow]"
                    )
                    console.print("[bold]Your choice:[/bold] ", end="")
                except Exception as e:
                    console.print(f"\n[red]Failed to open editor: {e}[/red]")
                    console.print(
                        f"[yellow]Please edit {temp_config_path} manually and type 'start' when ready.[/yellow]"  # noqa: E501
                    )
                    console.print("[bold]Your choice:[/bold] ", end="")
            elif choice == "cancel":
                console.print("[red]Training cancelled[/red]")
                try:
                    import typer

                    if yes:
                        # Always save in yes mode
                        config_path = Path("rl_config.json")
                        save_config(config, config_path)
                        hud_console.info("Auto-saved configuration (--yes mode)")
                    elif typer.confirm("Save this configuration for later?", default=True):
                        config_path = Path("rl_config.json")
                        save_config(config, config_path)
                except Exception as e:
                    hud_console.warning(f"Failed to save config: {e}")

                try:
                    temp_config_path.unlink()
                except Exception as e:
                    hud_console.warning(f"Failed to clean up temp config: {e}")

                try:
                    import typer

                    raise typer.Exit(0)
                except Exception:
                    return
            else:
                console.print(
                    "[red]Invalid choice. Type 'start', 'edit', or 'cancel':[/red] ", end=""
                )
    elif yes:
        # In yes mode, auto-start training
        hud_console.info("Auto-starting training (--yes mode)")
        config = load_config(temp_config_path)
    else:
        console.print("\n[dim]Using provided configuration file...[/dim]")
        config = load_config(temp_config_path)

    # Step 7: Determine if DDP should be used (imports heavy helpers lazily)
    num_gpus = len(gpu_info["devices"])
    use_ddp = False
    training_gpus = [0]  # Default single GPU
    vllm_gpu_idx = 1 if num_gpus > 1 else 0

    if num_gpus > 2 and not no_ddp:
        console.print(f"\n[cyan]🚀 Detected {num_gpus} GPUs - checking DDP configuration...[/cyan]")

        from .gpu_utils import calculate_optimal_gpu_allocation  # heavy import (torch at module)

        gpu_allocation = calculate_optimal_gpu_allocation(gpu_info, config)

        if gpu_allocation["use_ddp"]:
            use_ddp = True
            training_gpus = gpu_allocation["training_gpus"]
            vllm_gpu_idx = gpu_allocation["vllm_gpu"]

            console.print(
                f"[green]✅ Will use DDP with {len(training_gpus)} GPUs for training[/green]"
            )
            console.print(f"[green]✅ GPU {vllm_gpu_idx} reserved for vLLM server[/green]")

            console.print("\n[cyan]Training Configuration:[/cyan]")
            console.print(f"  • Groups to process: {gpu_allocation['num_groups']}")
            console.print(f"  • Training GPUs: {training_gpus}")
            console.print(f"  • Groups per GPU: {gpu_allocation.get('groups_per_gpu', 'N/A'):.1f}")

            if gpu_allocation.get("parallel_efficiency", 1.0) < 0.8:
                console.print(
                    f"\n[yellow]⚠️  GPU efficiency: {gpu_allocation['parallel_efficiency'] * 100:.0f}%[/yellow]"  # noqa: E501
                )
                console.print(
                    f"[yellow]Consider adjusting batch_size to {len(training_gpus) * config.training.group_size} for optimal performance[/yellow]"  # noqa: E501
                )
        else:
            console.print(f"[cyan]{gpu_allocation.get('reason', 'Using single GPU')}[/cyan]")

    # Allow manual overrides
    if ddp_gpus is not None:
        requested_gpus = [int(x) for x in ddp_gpus.split(",")]
        console.print(f"[cyan]Manual GPU selection: {requested_gpus}[/cyan]")
        available_indices = [d["index"] for d in gpu_info["devices"]]
        invalid_gpus = [g for g in requested_gpus if g not in available_indices]
        if invalid_gpus:
            console.print(f"[red]❌ Invalid/unhealthy GPU(s) requested: {invalid_gpus}[/red]")
            console.print(f"[yellow]Available healthy GPUs: {available_indices}[/yellow]")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return
        training_gpus = requested_gpus
        use_ddp = len(training_gpus) > 1

    if vllm_gpu is not None:
        vllm_gpu_idx = vllm_gpu
        console.print(f"[cyan]Manual vLLM GPU: {vllm_gpu_idx}[/cyan]")
        available_indices = [d["index"] for d in gpu_info["devices"]]
        if vllm_gpu_idx not in available_indices:
            console.print(f"[red]❌ vLLM GPU {vllm_gpu_idx} is not available/healthy![/red]")
            console.print(f"[yellow]Available healthy GPUs: {available_indices}[/yellow]")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return

    # Ensure we have at least one training GPU
    if not training_gpus:
        console.print("[red]❌ No available GPUs for training![/red]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return

    # Always adjust batch_size based on number of training GPUs (lazy import)
    from .gpu_utils import adjust_config_for_ddp  # heavy import (torch at module)

    config = adjust_config_for_ddp(config, len(training_gpus))
    save_config(config, temp_config_path)

    # Step 8: Start vLLM server (unless we're using a remote one)
    if not skip_vllm_startup:
        console.print(f"\n[cyan]Setting up vLLM server on GPU {vllm_gpu_idx}...[/cyan]")

        from .vllm import start_vllm_server, wait_for_vllm_server

        start_vllm_server(config.model.base_model, vllm_gpu_idx, restart=restart)
        server_ready = asyncio.run(wait_for_vllm_server())
        if not server_ready:
            console.print("[red]❌ Failed to start vLLM server[/red]")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return
    else:
        console.print("\n[cyan]Using remote vLLM server (skipping local startup)[/cyan]")

    # Step 9: Run training (DDP or single GPU)
    if use_ddp:
        console.print(
            f"\n[bold green]🎯 Starting DDP training on {len(training_gpus)} GPUs...[/bold green]\n"
        )
        launch_ddp_training(training_gpus, tasks_file, temp_config_path, verbose)
    else:
        console.print("\n[bold green]🎯 Starting single-GPU training...[/bold green]\n")
        try:
            # Set verbose in config instead of passing as parameter
            if verbose:
                config.verbose = True

            # Import and run the async training function lazily
            from hud.rl.train import train  # heavy import

            asyncio.run(train(config, tasks))  # type: ignore
            console.print("\n[green]✅ Training completed successfully![/green]")

            try:
                temp_config_path.unlink()
            except Exception as e:
                hud_console.warning(f"Failed to clean up temp config: {e}")

        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return
        except Exception as e:
            console.print(f"\n[red]❌ Training failed: {e}")
            try:
                import typer

                raise typer.Exit(1)
            except Exception:
                return


def launch_ddp_training(
    training_gpus: list[int], tasks_file: str, config_path: Path, verbose: bool
) -> None:
    """Launch DDP training with torchrun.

    Uses subprocess to run the training module, so heavy dependencies load in
    the spawned processes rather than the CLI import path.
    """
    import subprocess as _subprocess
    import sys as _sys

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, training_gpus))

    if not verbose:
        env["HUD_LOG_LEVEL"] = "WARNING"

    cmd = [
        _sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(training_gpus)}",
        "--master_port=29500",
        "-m",
        "hud.rl.train",
        "--config",
        str(config_path),
        "--tasks",
        tasks_file,
    ]

    if verbose:
        cmd.append("--verbose")

    try:
        _subprocess.run(cmd, env=env, check=True)  # noqa: S603
    except _subprocess.CalledProcessError as e:
        console.print(f"\n[red]❌ DDP training failed with exit code {e.returncode}[/red]")
        try:
            import typer

            raise typer.Exit(1)
        except Exception:
            return
    finally:
        try:
            config_path.unlink()
        except Exception as e:
            hud_console.warning(f"Failed to clean up temp config: {e}")
