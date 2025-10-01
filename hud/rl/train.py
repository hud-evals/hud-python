"""Main training loop for GRPO RL."""

from __future__ import annotations

import os

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hud
from hud.rl.actor import Actor
from hud.rl.buffer import DatasetBuffer, ReplayBuffer
from hud.rl.config import Config
from hud.rl.distributed import (
    broadcast_object,
    cleanup_distributed,
    get_global_rank,
    get_world_size,
    is_main_process,
    scatter_object,
    setup_distributed,
    synchronize,
)
from hud.rl.learner import GRPOLearner
from hud.rl.utils import (
    aggregate_metrics_across_ranks,
    ensure_dir,
    preprocess_advantages,
    set_seed,
)
from hud.rl.vllm_adapter import VLLMAdapter
from hud.utils.hud_console import HUDConsole
from hud.utils.tasks import load_tasks

if TYPE_CHECKING:
    from hud.types import Task
hud_console = HUDConsole(logging.getLogger(__name__))


async def train(config: Config, tasks: list[Task]) -> None:
    """Main training loop."""
    # Setup distributed environment
    setup_distributed()

    # Initialize components
    set_seed(config.seed + get_global_rank())  # Different seed per rank
    ensure_dir(config.out_dir)
    if config.verbose:
        logging.basicConfig(level=logging.INFO)
        # Remove httpx logger
        logging.getLogger("httpx").setLevel(logging.WARNING)
    if config.very_verbose:
        logging.basicConfig(level=logging.DEBUG)
        # Remove httpx logger
        logging.getLogger("httpx").setLevel(logging.INFO)

    if is_main_process():
        hud_console.header("Starting GRPO Training")
        hud_console.section_title(
            f"\n[1/3] Initializing components (world_size={get_world_size()})..."
        )

    num_gpus = get_world_size()

    # Actor is responsible for running tasks and collecting episodes
    actor = Actor(config) if is_main_process() else None

    # Learner is responsible for updating the policy
    learner = GRPOLearner(config)

    # Dataset buffer is responsible for storing tasks
    dataset_buffer = DatasetBuffer(tasks, config)
    if is_main_process():
        hud_console.key_value_table(dataset_buffer.info)

    if dataset_buffer.groups_per_batch % num_gpus != 0:
        hud_console.warning(
            f"Groups per batch {dataset_buffer.groups_per_batch} is not divisible by number of GPUs {num_gpus}"  # noqa: E501
        )
        exit(1)

    # Replay buffer is responsible for storing episodes for training
    trace_buffer = ReplayBuffer(config)

    # VLLM adapter is responsible for loading and unloading adapters (only on main process)
    vllm = (
        VLLMAdapter(config.actor.vllm_base_url, config.actor.vllm_api_key)
        if is_main_process()
        else None
    )

    # Load initial adapter if provided
    if is_main_process() and config.model.adapter_path and vllm:
        hud_console.info(f"Loading baseline adapter from: {config.model.adapter_path}")
        success = vllm.load_adapter(config.model.base_model, config.model.adapter_path)
        if success and actor is not None:
            hud_console.info("Successfully loaded baseline adapter as 'base_model'")
            # Update actor to use the loaded adapter
            actor.update_adapter(config.model.base_model)
        else:
            hud_console.error("Failed to load baseline adapter")
            exit(1)

    # Training state
    step = 0
    last_metrics = None  # Store last successful metrics for error recovery

    if is_main_process():
        hud_console.section_title("\n[2/3] Running training loop...")

    # Create job on main process and distribute ID across GPUs
    if is_main_process():
        hud_console.info(f"Creating job with config.job_id: {config.job_id}")
        job_obj = hud.create_job(
            job_id=config.job_id,
            name=config.job_name,
            metadata={"config": config.to_dict(), "agent_class": config.model.base_model},
        )
        hud_console.info(f"Created job with job_obj.id: {job_obj.id}")
        job_obj.update_status_sync("running")
        job_id = job_obj.id
    else:
        job_obj = None
        job_id = None

    # Broadcast job ID to all ranks
    job_id = broadcast_object(job_id, src=0)

    try:
        while len(dataset_buffer) > 0:
            if is_main_process():
                hud_console.section_title(f"Step {step + 1}/{dataset_buffer.training_steps}")
                hud_console.info(f"{len(dataset_buffer)} tasks remaining")
            # Get batch of tasks (all ranks need same tasks)
            tasks = dataset_buffer.get_tasks()

            # Initialize variables on all ranks
            global_reward_stats = None
            global_advantage_stats = None

            # Step-state gate: ensure all ranks branch coherently
            state = {"ok": False, "err": None, "num_samples": 0}
            rank_samples = None
            episode_time_value = None

            # Only rank 0 runs tasks and prepares distribution
            if is_main_process() and actor is not None:
                import time

                try:
                    episode_start_time = time.time()
                    traces = await actor.run_tasks(tasks, job_id=job_id)
                    episode_time = time.time() - episode_start_time
                    hud_console.info(f"Sampled {len(traces)} traces in {episode_time:.1f}s")
                    trace_buffer.add(traces)
                    global_reward_stats = [trace.reward for trace in traces]

                    # Get all traces from buffer for distribution
                    all_traces = trace_buffer.sample_traces()

                    # Preprocess traces to training samples
                    preprocessed_traces = preprocess_advantages(all_traces, config)

                    # Store these for later use in metrics
                    global_advantage_stats = [sample.advantage for sample in preprocessed_traces]

                    # Distribute preprocessed samples in groups across ranks via scatter
                    # Ensure list length is a multiple of num_gpus by allowing empty per-rank slices
                    gpu_batch_size = max(1, (len(preprocessed_traces) + num_gpus - 1) // num_gpus)
                    rank_samples = [
                        preprocessed_traces[i : i + gpu_batch_size]
                        for i in range(0, len(preprocessed_traces), gpu_batch_size)
                    ]
                    # Pad rank_samples to exactly num_gpus entries
                    if len(rank_samples) < num_gpus:
                        rank_samples.extend([[] for _ in range(num_gpus - len(rank_samples))])

                    # Log distribution info
                    dist_msg = (
                        f"Distributing {len(preprocessed_traces)} samples as {gpu_batch_size} "
                        f"sized batches across {num_gpus} GPUs"
                    )
                    hud_console.info(dist_msg)
                    for rank in range(num_gpus):
                        n_samples = len(rank_samples[rank]) if rank < len(rank_samples) else 0
                        hud_console.info(f"  Rank {rank}: {n_samples} samples")

                    hud_console.section_title(f"Training on {len(all_traces)} traces")
                    episode_time_value = episode_time

                    state.update({"ok": True, "num_samples": len(preprocessed_traces)})
                except Exception as e:
                    state.update({"ok": False, "err": str(e)})

            # Broadcast step-state to keep ranks in lockstep
            state = broadcast_object(state, src=0)
            if not state.get("ok", False):
                hud_console.warning("Step failed on rank 0; skipping this step coherently")
                synchronize()
                continue

            # Scatter per-rank samples; each rank receives only its slice
            my_samples = scatter_object(rank_samples if is_main_process() else None, src=0)
            # Broadcast the episode time (small object)
            episode_time_value = broadcast_object(episode_time_value, src=0)

            # Process only assigned samples
            last_metrics = learner.update(my_samples)

            # Add episode time (same for all ranks since episodes run on rank 0)
            if episode_time_value is not None:
                last_metrics.update(
                    {
                        "episode_time": episode_time_value,
                    }
                )

            # Aggregate metrics across all GPUs for proper statistics
            aggregate_metrics_across_ranks(last_metrics)

            if is_main_process() and job_obj is not None:
                # Use the global statistics we collected before distribution
                if global_reward_stats is not None and global_advantage_stats is not None:
                    last_metrics.update(
                        {
                            "advantage": global_advantage_stats,
                            "reward": global_reward_stats,
                        }
                    )
                else:
                    # Fallback: use only this rank's data
                    hud_console.warning("Global statistics not available, using partial data")
                    last_metrics.update(
                        {
                            "advantage": [sample.advantage for sample in my_samples]
                            if my_samples
                            else [],
                            "reward": [sample.reward for sample in my_samples]
                            if my_samples
                            else [],
                        }
                    )

                job_obj.log_sync(last_metrics.to_dict())

                if step % config.stats_interval == 0:
                    hud_console.key_value_table(last_metrics.to_dict())

            # Increment step counter on all processes
            step += 1

            # Save checkpoint and update vLLM (only on main process)
            if step % config.training.save_every_batches == 0:
                if is_main_process() and vllm is not None and actor is not None:
                    hud_console.section_title("Saving checkpoint and updating vLLM")
                    checkpoint_path = Path(config.out_dir) / f"{config.adapter_prefix}-{step}"
                    learner.save(str(checkpoint_path))

                    # Wait for 6 seconds to ensure the checkpoint is saved
                    await asyncio.sleep(6)

                    # If there is a previous adapter, unload it
                    current_adapter = vllm.get_current()
                    if current_adapter is not None:
                        vllm.unload_adapter(current_adapter)

                    adapter_name = f"{config.adapter_prefix}-{step}"
                    if vllm.load_adapter(adapter_name, str(checkpoint_path)):
                        actor.update_adapter(adapter_name)
                        hud_console.info(f"✓ Checkpoint saved and loaded: {adapter_name}")
                    else:
                        hud_console.warning(f"Failed to hot-load adapter {adapter_name}")

                # Ensure all processes wait for checkpoint operations to complete
                synchronize()

        if is_main_process():
            hud_console.section_title("\n[3/3] Training completed!")
            # Update job status to completed
            if job_obj:
                job_obj.update_status_sync("completed")
    except Exception as e:
        # Log error and any available metrics before failing
        hud_console.error(f"Training failed on rank {get_global_rank()}: {e}")

        if is_main_process():
            # Log final metrics if we have any
            if last_metrics and job_obj:
                try:
                    job_obj.log_sync(last_metrics.to_dict())
                except Exception:
                    hud_console.warning("Failed to log final metrics")

            # Update job status to failed
            if job_obj:
                job_obj.update_status_sync("failed")

        # Don't re-raise immediately to allow cleanup
        raise

    finally:
        # Try to sync one last time, but don't fail if it doesn't work
        try:
            synchronize()
        except Exception:
            hud_console.warning("Failed to synchronize during cleanup")

        # Clean up distributed environment
        cleanup_distributed()


async def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO RL Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    # Task input arguments
    parser.add_argument(
        "--tasks", type=str, help="Path to tasks JSONL file or HuggingFace dataset name"
    )
    parser.add_argument("--tasks-json", type=json.loads, help="Tasks as JSON list string")

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, encoding="utf-8") as f:  # noqa: ASYNC230
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()

    # Apply test mode settings
    if args.test:
        hud_console.info("[TEST MODE] Using minimal configuration")
        eps = 6
        config.training.batch_size = eps
        config.actor.max_parallel_episodes = 12
        config.training.group_size = eps
        config.training.mini_batch_size = 3
        config.training.training_steps = 4
        config.actor.max_steps_per_episode = 4

    # Calculate the memory usage
    INITIAL_MEMORY = 8.0
    SCALING_FACTOR = 4 / (28 * 28 * 256 * 1024)
    token_estimate = (
        config.training.mini_batch_size
        * config.actor.max_steps_per_episode
        * config.actor.max_new_tokens
    )
    hud_console.info(f"Estimated tokens per forward pass: {token_estimate}")
    image_estimate = config.model.max_pixels
    total_memory = INITIAL_MEMORY + SCALING_FACTOR * token_estimate * image_estimate
    hud_console.info(f"Estimated memory peak: {total_memory:.2f} GB")
    if total_memory > 75.0:
        hud_console.warning(
            "Potential memory usage is too high, decrease either training steps or mini batch size"
        )
        exit(1)

    # Load tasks
    if args.tasks_json:
        # Tasks provided as JSON list via command line
        tasks = load_tasks(args.tasks_json)
    elif args.tasks:
        # Tasks provided as file path or HuggingFace dataset
        tasks = load_tasks(args.tasks)
    else:
        # Default to browser_2048_tasks.jsonl if it exists
        default_tasks_path = "browser_2048_tasks.jsonl"
        if Path(default_tasks_path).exists():
            hud_console.info(f"No tasks specified, using default: {default_tasks_path}")
            tasks = load_tasks(default_tasks_path)
        else:
            raise ValueError(
                "No tasks specified. Use --tasks, --tasks-json, or specify tasks_file in config"
            )

    # Run training
    tasks_typed = cast("list[Task]", tasks)
    await train(config, tasks_typed)


if __name__ == "__main__":
    asyncio.run(main())
