from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
 

from torch.distributed import get_rank

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from hud.rl.config import Config, TrainingConfig
from hud.rl.logger import console, configure_logging
from hud.rl.loss import compute_loss, get_per_token_logps, entropy_from_logits
from hud.rl.metrics import MetricsCollector
from hud.rl.model import build_model
from hud.rl.optimizer import get_optimizer
from hud.rl.parallel_dims import ParallelDims
from hud.rl.utils import get_world_size, setup_distributed, is_main_process
from hud.rl.checkpoint import CheckpointManager
from hud.rl.utils import save_step_metrics
from hud.rl.types import TrainingSample
from hud.rl.perf import PerfCounter
from rich.table import Table


def get_batch(step: int, root: str) -> list[TrainingSample]:
    """Load the batch for the given step from ``<output_dir>/step_{step}/rollouts``."""

    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    output_root = Path(root)
    step_rollout_dir = output_root / f"step_{step:05d}" / "rollouts"
    batch_path = step_rollout_dir / f"rank_{rank}.pt"

    console.info(f"Waiting for batch: {batch_path}")
    while not batch_path.exists():
        time.sleep(0.5)

    console.info_log(f"Loading batch from {batch_path}")
    return torch.load(batch_path, weights_only=False)

def train(
    training_config: TrainingConfig,
    max_steps: int,
) -> None:

    setup_distributed()
    world_size = get_world_size()
    rank = get_rank()

    console.section_title("Initializing trainer")

    if training_config.benchmark:
        if is_main_process():
            console.warning_log("Running in benchmark mode, overriding max_steps to 5")
        max_steps = min(max_steps, 5)


    parallel_dims = ParallelDims(
        dp_replicate=training_config.dp_replicate,
        dp_shard=training_config.dp_shard,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    model = build_model(training_config, parallel_dims)

    benchmark_data = []

    ref_model: torch.nn.Module | None = None
    if training_config.loss.kl_beta > 0:
        console.info_log("Initializing reference model for KL regularization")
        ref_model = build_model(training_config, parallel_dims)
        ref_model.eval()

    optimizer = get_optimizer(training_config.optimizer, model)

    checkpoint_manager = CheckpointManager(
        output_dir=training_config.output_dir,
        save_last_n=training_config.save_last_n,
    )

    collector = MetricsCollector(distributed=(world_size > 1))

    perf_counter: PerfCounter | None = None

    for step in range(max_steps):
        collector.reset()
        # Save checkpoint from previous step (skip first step since no training yet)
        if step > 0:
            console.info(f"Saving checkpoint for step {step - 1}...")
            checkpoint_manager.save(model, step - 1)
        
        batch = get_batch(step, training_config.output_dir)
                
        if ref_model is not None:
            with console.progress("Computing reference log probabilities...") as progress, torch.no_grad():
                for i, minibatch in enumerate(batch):
                    sample = minibatch.to_device(torch.device("cuda"))
                    logits = ref_model(**sample.inputs).logits
                    logits = torch.cat(
                        [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                    )
                    logits = logits / sample.temperature.view(-1, 1, 1)
                    sample.ref_logprobs = get_per_token_logps(
                        logits,
                        sample.inputs["input_ids"],
                    ).cpu()
                    del logits
                    progress.update(f"Computing reference log probabilities... {i + 1}/{len(batch)}")

        if perf_counter is None:
            perf_counter = PerfCounter(model, batch[0].inputs["input_ids"].shape[1], 10)
            perf_counter.count_tokens(0)

        with console.progress("Computing old log probabilities...") as progress, torch.no_grad():
            for i, minibatch in enumerate(batch):
                sample = minibatch.to_device(torch.device("cuda"))
                logits = model(**sample.inputs).logits
                logits = torch.cat(
                    [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                )
                logits = logits / sample.temperature.view(-1, 1, 1)
                sample.old_logprobs = get_per_token_logps(
                    logits,
                    sample.inputs["input_ids"],
                ).cpu()
                del logits
                progress.update(f"Computing old log probabilities... {i + 1}/{len(batch)}")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        training_start_time = time.time()

        if training_config.loss.importance_sampling_level == "token":
            loss_norm = int(sum(
                minibatch.inputs["assistant_mask"].sum().item() for minibatch in batch
            ))
        elif training_config.loss.importance_sampling_level == "sequence":
            loss_norm = len(batch)

        with console.progress("Training...") as progress:
            for idx, minibatch in enumerate(batch):
                model.set_requires_all_reduce(idx == len(batch) - 1)

                sample = minibatch.to_device(torch.device("cuda"))
                logits = model(**sample.inputs).logits
                logits = torch.cat(
                    [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
                )
                logits = logits / sample.temperature.view(-1, 1, 1)
                logprobs = get_per_token_logps(
                    logits,
                    sample.inputs["input_ids"],
                )

                # Compute entropy for masked tokens in chunks
                with torch.no_grad():
                    mask = sample.inputs["assistant_mask"].bool()
                    if mask.any():
                        ent = entropy_from_logits(logits, mask, chunk_size=128)
                        collector.log(entropy=ent)

                # old_logprobs is set in the previous loop, so it should not be None
                assert sample.old_logprobs is not None, "old_logprobs must be computed before training"

                loss, loss_tensors = compute_loss(
                    logprobs,
                    sample.old_logprobs,
                    sample.ref_logprobs,
                    sample.advantage,
                    sample.inputs["assistant_mask"],
                    training_config.loss,
                    loss_norm,
                )

                collector.log(loss=loss.detach().cpu())

                mask = sample.inputs["assistant_mask"].bool()
                for name, tensor in loss_tensors.items():
                    if isinstance(tensor, torch.Tensor) and tensor.shape == mask.shape:
                        collector.log(**{name: tensor[mask]})
                
                del logits
                loss.backward()
                progress.update(f"Trained... {idx + 1}/{len(batch)}")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.max_grad_norm
        )
        collector.log(grad_norm=grad_norm.detach())

        optimizer.step()
        optimizer.zero_grad()
        
        step_duration = time.time() - training_start_time
        console.info_log(f"Step {step} training took {step_duration:.2f} seconds")


        # Collect performance data
        # sum batch size * sequence length for each minibatch
        num_tokens = sum(minibatch.inputs["input_ids"].shape[1] * minibatch.inputs["input_ids"].shape[0] for minibatch in batch)
        console.warning_log(f"num_tokens: {num_tokens}")
        perf_counter.count_tokens(num_tokens)  # Add to rolling window
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        dist_perf_output_list = [{}] * world_size
        torch.distributed.all_gather_object(dist_perf_output_list, {
            "step_duration": step_duration,
            "throughput": throughput,
            "mfu": mfu,
            "peak_memory": peak_memory,
        })

        benchmark_data.append({
            "step": step,
            # max step duration across ranks
            "step_duration": max([x["step_duration"] for x in dist_perf_output_list]),
            # sum throughput across ranks
            "throughput": sum([x["throughput"] for x in dist_perf_output_list]),
            # sum mfu across ranks (already normalized by world size)
            "mfu": sum([x["mfu"] for x in dist_perf_output_list]),
            # sum peak memory across ranks
            "peak_memory": max([x["peak_memory"] for x in dist_perf_output_list]),
        })

        stats = collector.get_stats()
        if is_main_process():
            save_step_metrics(training_config.output_dir, step, stats)

        torch.cuda.empty_cache()
    
    # Save final checkpoint after last training step
    if max_steps > 0:
        console.info(f"Saving final checkpoint for step {max_steps - 1}...")
        checkpoint_manager.save(model, max_steps - 1)

    if training_config.benchmark:
        # Create benchmark table
        table = Table(title="Training Performance Metrics")
        table.add_column("Step", justify="right", style="cyan")
        table.add_column("Duration (s)", justify="right", style="yellow")
        table.add_column("Throughput (tok/s)", justify="right", style="green")
        table.add_column("MFU (%)", justify="right", style="green")
        table.add_column("Peak Memory (GB)", justify="right", style="red")
        
        # Add rows
        for data in benchmark_data:
            table.add_row(
                str(data["step"]),
                f"{data['step_duration']:.2f}",
                f"{data['throughput']:.0f}",
                f"{data['mfu']:.2f}",
                f"{data['peak_memory']:.2f}",
            )
        
        if is_main_process():
            console.section_title("Benchmark Results")
            console.print(table)
        


def main() -> None:
    """Main entry point for training script."""
    config, _ = Config.from_argv()
    configure_logging(config.verbose)
    train(config.training, config.training_steps)


if __name__ == "__main__":
    main()
