import os

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import hud
from hud.rl.actor import Actor
from hud.rl.buffer import ReplayBuffer, SimpleBuffer
from hud.rl.checkpoint import CheckpointManager
from hud.rl.config import Config
from hud.rl.dataloader import DataLoader
from hud.rl.distributed import (
    broadcast_object,
    cleanup_distributed,
    get_global_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    synchronize,
)
from hud.rl.logger import console
from hud.rl.loss import compute_grpo_loss, compute_logprobs, sanity_check
from hud.rl.metrics import MetricsCollector
from hud.rl.model import load_models
from hud.rl.optimizer import create_optimizer
from hud.rl.utils import (
    ensure_dir,
    get_gpu_utilization,
    get_memory_usage,
    set_seed,
)
from hud.rl.vllm_adapter import VLLMAdapter
from hud.utils.tasks import load_tasks

if TYPE_CHECKING:
    from hud.types import Task


def prepare_groups(samples: list[Any], policy: Any, config: Config) -> list[list[Any]]:
    """Prepare groups of samples for training by computing logprobs."""
    batch = samples

    with console.progress("Computing logprobs for batch...") as progress, torch.no_grad():
        for i, sample in enumerate(batch):
            if is_main_process():
                progress.update(f"Processing batch of traces... {i}/{len(batch)}")
            if sample.inputs:
                sample = sample.to_device(torch.device(f"cuda:{get_global_rank()}"))
                sample.old_logprobs, _ = compute_logprobs(
                    policy, sample.inputs, temperature=config.actor.temperature
                )

        policy_module = policy.module if hasattr(policy, "module") else policy
        with policy_module.disable_adapter():
            for i, sample in enumerate(batch):
                if is_main_process():
                    progress.update(f"Processing reference logprobs... {i}/{len(batch)}")
                if sample.inputs:
                    sample.ref_logprobs, _ = compute_logprobs(
                        policy, sample.inputs, temperature=config.actor.temperature
                    )

    for sample in batch:
        sample.to_device(torch.device("cpu"))

    # Convert to grouped batches based on configuration
    group_size = config.training.group_size
    if not config.training.accumulate_over_minibatches:
        group_size = group_size // config.training.mini_batch_size

    if config.training.update_after_group:
        return [batch[i : i + group_size] for i in range(0, len(batch), group_size)]
    else:
        return [batch]


def compute_loss(sample: Any, policy: Any, config: Config, collector: MetricsCollector) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for a batch of samples."""
    device = torch.device(f"cuda:{get_global_rank()}")
    sample.to_device(device)

    pol_logp, pol_entropy = compute_logprobs(
        policy,
        sample.inputs,
        temperature=config.actor.temperature,
    )

    sanity_check(sample, pol_logp, sample.old_logprobs, sample.ref_logprobs)

    console.info_log(
        f"GPU Util: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB"
    )

    total_loss, metrics_dict = compute_grpo_loss(
        sample=sample,
        pol_logp=pol_logp,
        pol_entropy=pol_entropy,
        old_logp=sample.old_logprobs,
        ref_logp=sample.ref_logprobs,
        config=config,
    )

    sample.to_device(torch.device("cpu"))

    return total_loss, metrics_dict


def update_model(
    samples: list[Any], policy: Any, optimizer: Any, config: Config, collector: MetricsCollector
) -> None:
    """Perform a gradient update on a batch."""
    import time
    from contextlib import nullcontext

    training_start_time = time.time()

    # Prepare groups for GRPO training
    groups = prepare_groups(samples, policy, config)
    console.info_log(f"Updating over {len(groups)} groups")

    # Update over mini batch size
    with console.progress("Gradient update...") as progress:
        for epoch in range(config.training.epochs):  # Do not accumulate across epochs
            progress.update(f"Training epoch {epoch + 1}/{config.training.epochs}")
            for group_idx, group in enumerate(groups):  # Do not accumulate across "groups"
                optimizer.zero_grad(set_to_none=True)

                debug_per_group = ""
                grad_accum_steps = len(group)
                # Tensor for distributed sync
                device = torch.device(f"cuda:{get_global_rank()}")
                global_skip = torch.zeros(1, device=device)

                for s_idx, sample_minibatch in enumerate(group):
                    # Do not sync until the last minibatch
                    if s_idx < len(group) - 1 and get_world_size() > 1:
                        ddp_ctx = policy.no_sync()
                    else:
                        ddp_ctx = nullcontext()

                    with ddp_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        try:
                            loss, loss_metrics = compute_loss(sample_minibatch, policy, config, collector)
                            loss = loss / grad_accum_steps
                            collector.log(**loss_metrics)
                            debug_per_group += f"l{s_idx}:{round(loss.item(), 3)!s} "
                            loss.backward()
                        except torch.cuda.OutOfMemoryError:
                            console.warning_log(
                                f"{group_idx} CUDA OOM for {sample_minibatch.inputs['input_ids'].numel()} tokens; skipping minibatch"
                            )
                            # Dummy backward to keep DDP happy
                            dummy = torch.sum(p.sum() for p in policy.parameters()) * 0.0
                            debug_per_group += f"o{s_idx}:{round(dummy.item(), 3)!s} "
                            dummy.backward()
                            # mark global skip if OOM
                            global_skip.fill_(1)
                            continue

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # After minibatches loop, sync skip across ranks
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_skip, op=torch.distributed.ReduceOp.MAX)
                    skip_any = bool(global_skip.item())

                    if skip_any:
                        console.info_log(f"G[{group_idx}] {debug_per_group} N/A (skipped)")
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        config.training.grad_clip,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()

                    debug_per_group += f"g:{round(grad_norm.item(), 3)!s}"
                    console.info_log(f"G[{group_idx}] {debug_per_group}")

                    collector.log(
                        grad_norm=grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else float(grad_norm)
                    )

    # Calculate training time and throughput
    training_time = time.time() - training_start_time
    total_samples = len(groups) * config.training.group_size * config.training.mini_batch_size
    samples_per_second = total_samples / training_time if training_time > 0 else 0.0

    collector.log(
        training_time=training_time,
        samples_per_second=samples_per_second
    )


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

    if is_main_process():
        console.header("Starting GRPO Training")
        console.section_title(f"\n[1/3] Initializing components (world_size={get_world_size()})...")

    num_gpus = get_world_size()

    # Actor is responsible for running tasks and collecting episodes
    actor = Actor(config) if is_main_process() else None

    # Load models and optimizer directly
    processor, policy, ref = load_models(config)
    optimizer = create_optimizer(config, policy)

    # Create checkpoint manager for training loop
    checkpoint_manager = CheckpointManager(
        out_dir=config.out_dir, adapter_prefix=config.adapter_prefix
    )

    # Initialize metrics collector
    collector = MetricsCollector(distributed=config.distributed.enabled)

    # Initialize buffer based on select strategy
    if config.training.select_strategy in ["variance", "random"]:
        buffer = ReplayBuffer(tasks, config)
    else:
        buffer = SimpleBuffer(tasks, config)

    if is_main_process():
        console.key_value_table(buffer.info)

    if buffer.groups_per_batch % num_gpus != 0:
        console.warning(
            f"Groups per batch {buffer.groups_per_batch} is not divisible by number of GPUs {num_gpus}"  # noqa: E501
        )
        exit(1)

    # Initialize dataloader with buffer
    dataloader = DataLoader(buffer, config, processor=processor, policy=policy)

    # VLLM adapter is responsible for loading and unloading adapters (only on main process)
    vllm = (
        VLLMAdapter(config.actor.vllm_base_url, config.actor.vllm_api_key)
        if is_main_process()
        else None
    )

    # Training state
    step = 0
    last_metrics = None  # Store last successful metrics for error recovery

    if is_main_process():
        console.section_title("\n[2/3] Running training loop...")

    # Create job on main process and distribute ID across GPUs
    if is_main_process():
        console.info(f"Creating job with config.job_id: {config.job_id}")
        job_obj = hud.create_job(
            job_id=config.job_id, name=config.job_name, metadata={"config": config.to_dict()}
        )
        console.info(f"Created job with job_obj.id: {job_obj.id}")
        job_obj.update_status_sync("running")
        job_id = job_obj.id
    else:
        job_obj = None
        job_id = None

    # Broadcast job ID to all ranks
    job_id = broadcast_object(job_id, src=0)

    try:
        while len(buffer) > 0:
            if is_main_process():
                console.section_title(f"Step {step + 1}/{buffer.training_steps}")
                console.info(f"{len(buffer)} tasks remaining")
            # Get batch of tasks (all ranks need same tasks)
            tasks = buffer.sample_tasks()

            # Initialize variables on all ranks
            global_reward_stats = None
            global_advantage_stats = None

            # Only rank 0 runs tasks and collects traces
            if is_main_process() and actor is not None:
                import time

                episode_start_time = time.time()
                traces = await actor.run_tasks(tasks, job_id=job_id)
                episode_time = time.time() - episode_start_time
                console.info(f"Sampled {len(traces)} traces in {episode_time:.1f}s")
                buffer.add_traces(traces)
                global_reward_stats = [trace.reward for trace in traces]

                # Get preprocessed training batch from dataloader
                preprocessed_traces = dataloader.get_training_batch()

                # Store these for later use in metrics
                global_advantage_stats = [sample.advantage for sample in preprocessed_traces]

                # Distribute preprocessed samples in groups across ranks
                gpu_batch_size = len(preprocessed_traces) // num_gpus
                rank_samples = [
                    preprocessed_traces[i : i + gpu_batch_size]
                    for i in range(0, len(preprocessed_traces), gpu_batch_size)
                ]

                # Log distribution info
                console.info(
                    f"Distributing {len(preprocessed_traces)} samples as {gpu_batch_size} sized batches across {num_gpus} GPUs"  # noqa: E501
                )
                for rank in range(num_gpus):
                    n_samples = len(rank_samples[rank])
                    console.info(f"  Rank {rank}: {n_samples} samples")

                console.section_title(f"Training on {len(traces)} traces")
                episode_time_value = episode_time
            else:
                rank_samples = None
                episode_time_value = None

            # Broadcast each rank's samples and episode time
            rank_samples = broadcast_object(rank_samples, src=0)
            episode_time_value = broadcast_object(episode_time_value, src=0)
            my_samples = rank_samples[get_global_rank()] if rank_samples else []

            # Process only assigned samples
            update_model(my_samples, policy, optimizer, config, collector)

            # Add episode time (same for all ranks since episodes run on rank 0)
            if episode_time_value is not None:
                collector.log(episode_time=episode_time_value)

            # Get aggregated metrics (handles distributed aggregation internally)
            last_metrics = collector.get_stats()

            if is_main_process() and job_obj is not None:
                # Use the global statistics we collected before distribution
                if global_reward_stats is not None and global_advantage_stats is not None:
                    for adv in global_advantage_stats:
                        if adv is not None:
                            collector.log(advantage=adv)
                    for rew in global_reward_stats:
                        if rew is not None:
                            collector.log(reward=rew)
                else:
                    # Fallback: use only this rank's data
                    console.warning("Global statistics not available, using partial data")
                    for sample in my_samples:
                        if sample.advantage is not None:
                            collector.log(advantage=sample.advantage)
                        if sample.reward is not None:
                            collector.log(reward=sample.reward)

                # Get updated stats after adding reward/advantage
                last_metrics = collector.get_stats()

                job_obj.log_sync(last_metrics.to_dict())

                if step % config.stats_interval == 0:
                    console.key_value_table(last_metrics.to_dict())

            # Reset metrics for next iteration and increment step
            collector.reset()
            step += 1

            # Save checkpoint and update vLLM (only on main process)
            if step % config.training.save_every_batches == 0:
                if is_main_process() and vllm is not None and actor is not None:
                    console.section_title("Saving checkpoint and updating vLLM")

                    # Create checkpoint path with timestamp
                    checkpoint_path, adapter_name = checkpoint_manager.create_timestamped_path()
                    checkpoint_manager.save(policy, checkpoint_path)

                    # Wait for 6 seconds to ensure the checkpoint is saved
                    await asyncio.sleep(6)

                    if vllm.load_adapter(adapter_name, checkpoint_path):
                        actor.update_adapter(adapter_name)
                        console.info(f"✓ Checkpoint saved and loaded: {adapter_name}")
                    else:
                        console.warning(f"Failed to hot-load adapter {adapter_name}")

                # Ensure all processes wait for checkpoint operations to complete
                synchronize()

        if is_main_process():
            console.section_title("\n[3/3] Training completed!")
            # Update job status to completed
            if job_obj:
                job_obj.update_status_sync("completed")
    except Exception as e:
        # Log error and any available metrics before failing
        console.error(f"Training failed on rank {get_global_rank()}: {e}")

        if is_main_process():
            # Log final metrics if we have any
            if last_metrics and job_obj:
                try:
                    job_obj.log_sync(last_metrics.to_dict())
                except Exception:
                    console.warning("Failed to log final metrics")

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
            console.warning("Failed to synchronize during cleanup")

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
        with open(args.config) as f:  # noqa: ASYNC230
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()

    # Apply test mode settings
    if args.test:
        console.info("[TEST MODE] Using minimal configuration")
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
    console.info(f"Estimated tokens per forward pass: {token_estimate}")
    image_estimate = config.model.max_pixels
    total_memory = INITIAL_MEMORY + SCALING_FACTOR * token_estimate * image_estimate
    console.info(f"Estimated memory peak: {total_memory:.2f} GB")
    if total_memory > 75.0:
        console.warning(
            "Potential memory usage is too high, decrease either training steps or mini batch size"
        )
        exit(1)

    # Load tasks
    if args.tasks_json:
        # Tasks provided as JSON list via command line
        tasks = load_tasks(args.tasks_jso)
    elif args.tasks:
        # Tasks provided as file path or HuggingFace dataset
        tasks = load_tasks(args.tasks)
    else:
        # Default to browser_2048_tasks.jsonl if it exists
        default_tasks_path = "browser_2048_tasks.jsonl"
        if Path(default_tasks_path).exists():
            console.info(f"No tasks specified, using default: {default_tasks_path}")
            tasks = load_tasks(default_tasks_path)
        else:
            raise ValueError(
                "No tasks specified. Use --tasks, --tasks-json, or specify tasks_file in config"
            )

    # Run training
    await train(config, tasks)


if __name__ == "__main__":
    asyncio.run(main())
