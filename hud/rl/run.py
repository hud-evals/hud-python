from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import hud
from hud.rl.actor import Actor
from hud.rl.advantages import calculate_advantages
from hud.rl.batch import batch_samples
from hud.rl.buffer import create_buffer
from hud.rl.config import Config
from hud.rl.logger import console
from hud.rl.model import get_processor
from hud.rl.utils import get_weights_path
from hud.rl.vllm import update_weights
from openai import AsyncOpenAI
from hud.rl.preprocessor import preprocess_traces
from hud.rl.types import TrainingSample
from hud.utils.tasks import load_tasks

if TYPE_CHECKING:
    from hud.types import Task
    from transformers.processing_utils import ProcessorMixin

def resolve_pad_token_id(processor: "ProcessorMixin") -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)

def save_batches(training_batch: list[list[TrainingSample]], step: int, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    rollout_dir = output_dir / f"step_{step:05d}" / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    
    for rank, rank_samples in enumerate(training_batch):
        batch_path = rollout_dir / f"rank_{rank}.pt"
        temp_path = rollout_dir / f"rank_{rank}.pt.tmp"
        
        # Write to temp file first
        torch.save(rank_samples, temp_path)
        
        # Atomically rename to final path
        temp_path.rename(batch_path)
        console.debug_log(f"Saved {len(rank_samples)} minibatches to {batch_path}")
    
    console.info(f"Saved batches for step {step} to {rollout_dir}")

async def wait_for_checkpoint(step: int, output_dir: str | Path, timeout: int = 3600) -> Path:
    checkpoint_path = get_weights_path(output_dir, step)
    console.info(f"Waiting for checkpoint: {checkpoint_path}")
    
    start_time = time.time()
    while not checkpoint_path.exists():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Checkpoint not found after {timeout} seconds: {checkpoint_path}")
        
        await asyncio.sleep(0.5)
    
    console.info(f"Checkpoint ready: {checkpoint_path}")
    return checkpoint_path

async def run(config: Config, tasks: list[Task]) -> None:
    "Run Rollouts, Collect and Prepare Training Samples"

    client = AsyncOpenAI(
        base_url=config.client.base_url.replace("localhost", "127.0.0.1"),
        api_key=config.client.api_key,
        timeout=float(config.client.request_timeout),
    )

    # Actor constructs its own agent using the provided client
    actor = Actor(config.actor, client=client)
    buffer = create_buffer(
        tasks=tasks,
        group_size=config.group_size,
        select_strategy=config.buffer.select_strategy,
        buffer_steps=config.buffer.buffer_steps,
        shuffle_dataset=config.buffer.shuffle_dataset,
        require_images=config.buffer.require_images,
    )

    console.key_value_table(buffer.info)

    # Ensure enough unique tasks
    min_num_tasks= config.batch_size // config.group_size
    num_unique_tasks = len(set(task.id for task in tasks))
    
    if num_unique_tasks < min_num_tasks:
        raise ValueError(
            f"Insufficient unique tasks: need at least {min_num_tasks} "
            f"(batch_size={config.batch_size} / group_size={config.group_size}), "
            f"but only {num_unique_tasks} unique task(s) loaded. "
            f"The buffer will not be able to collect enough completed groups."
        )

    job_metadata = {
        "base_model": config.base_model,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "mini_batch_size": config.mini_batch_size,
        "buffer_steps": config.buffer.buffer_steps,
        "select_strategy": config.buffer.select_strategy,
        "training_steps": config.training_steps,
        "expected_world_size": config.expected_world_size,
        "grad_accumulation_steps": config.grad_accumulation_steps,
    }

    with hud.job(config.job_name, metadata=job_metadata, job_id=config.job_id) as job:
        processor = get_processor(config.base_model, config.processor)
        pad_token_id = resolve_pad_token_id(processor)

        num_tasks = config.batch_size // config.group_size
        console.info(f"Number of prompts per step: {num_tasks}")
        total_steps = config.training_steps

        console.section_title("Starting Run")
        for step in range(total_steps):
            console.section_title(f"Step {step + 1}/{total_steps}")

            # Collect traces until buffer has enough completed groups
            while buffer.completed_groups() < num_tasks:
                tasks_to_run = buffer.sample_tasks(num_tasks)

                console.info(f"Running {len(tasks_to_run)} tasks")
                collected_traces = await actor.run_tasks(tasks_to_run, job_id=job.id)

                # Count successful vs error traces
                successful = sum(1 for t in collected_traces if not getattr(t, "isError", False))
                console.info(
                    f"Collected {len(collected_traces)} traces ({successful} successful, {len(collected_traces)-successful} errors)"
                )

                buffer.add_traces(collected_traces)
                console.info(
                    f"Buffer status: {buffer.completed_groups()}/{num_tasks} groups complete"
                )

            # Sample traces for training
            traces = buffer.sample_traces(num_tasks)
            if not traces:
                console.warning("No traces sampled from buffer")
                buffer.reset()
                continue

            # Prepare rewards and advantages
            rewards = torch.tensor([float(trace.reward) for trace in traces], dtype=torch.float32)

            advantages = calculate_advantages(
                rewards=rewards,
                group_size=config.group_size,
                scale_rewards=config.rewards.scale_rewards,
                leave_one_out=config.rewards.leave_one_out,
            )

            temperatures = torch.tensor(
                [trace.info["temperature"] for trace in traces],
                dtype=torch.float32,
            )

            processed_inputs = preprocess_traces(traces, processor)

            samples: list[TrainingSample] = []
            for inputs, advantage, temperature in zip(
                processed_inputs,
                advantages,
                temperatures,
                strict=True,
            ):
                samples.append(
                    TrainingSample(
                        inputs=inputs,
                        advantage=advantage.view(1),
                        temperature=temperature.view(1),
                    )
                )

            training_batch = batch_samples(samples, config.mini_batch_size, config.num_gpus, pad_token_id)

            total_samples = sum(len(gpu_batch) for gpu_batch in training_batch)
            console.info(f"Created training batch with {total_samples} samples across {len(training_batch)} GPU batches")
            
            save_batches(training_batch, step, config.output_dir)
            
            console.section_title(f"Waiting for training to complete for step {step}")
            checkpoint_path = await wait_for_checkpoint(step, config.output_dir)
            
            if step < config.training_steps - 1:
                console.section_title(f"Updating vLLM weights from checkpoint: {checkpoint_path}")
                try:
                    await update_weights(client, Path(config.output_dir), step)
                    console.info("vLLM weights updated successfully")
                except Exception as e:
                    console.warning_log(f"Failed to update vLLM weights: {e}")
                    console.warning_log("Continuing with current weights...")
            else:
                console.info("Last step - skipping vLLM weight update")
            
            buffer.reset()
            console.info(f"Buffer reset. Status: {buffer.info}")

        console.section_title("All steps completed")


async def _main_async() -> None:
    import sys

    # Extract --tasks args before from_argv
    tasks_arg = None
    filtered_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--tasks" and i + 1 < len(sys.argv):
            tasks_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--tasks-json" and i + 1 < len(sys.argv):
            tasks_arg = json.loads(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    sys.argv = [sys.argv[0]] + filtered_argv
    config, _ = Config.from_argv()

    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    if not tasks_arg:
        raise ValueError("Requires tasks via --tasks or --tasks-json")

    tasks = load_tasks(tasks_arg)
    await run(config, tasks) # type: ignore


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
