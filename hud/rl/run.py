from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

async def run(config: Config, tasks: list[Task]) -> None:
    "Run Rollouts, Collect and Prepare Training Samples"

    # Initialize components
    actor = Actor(config.actor)
    buffer = create_buffer(
        tasks=tasks,
        group_size=config.group_size,
        select_strategy=config.select_strategy,
        buffer_steps=config.buffer_steps,
        shuffle_dataset=config.shuffle_dataset,
    )

    console.key_value_table(buffer.info)

    job_metadata = {
        "base_model": config.base_model,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "mini_batch_size": config.mini_batch_size,
        "buffer_steps": config.buffer_steps,
        "select_strategy": config.select_strategy,
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

            training_batch = batch_samples(samples, config.mini_batch_size, pad_token_id)

            # Reset buffer for next batch
            buffer.reset()
            console.info(f"Buffer reset. Status: {buffer.info}")

        console.section_title("All steps completed")


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="Run HUD RL")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--tasks", type=str, help="Path to tasks JSONL file or HuggingFace dataset name")
    parser.add_argument("--tasks-json", type=json.loads, help="Tasks as JSON list string")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.config:
        with open(args.config) as fp:
            config_dict = json.load(fp)
        config = Config.model_validate(config_dict)
    else:
        config = Config()

    if args.verbose:
        config.verbose = True
        logging.basicConfig(level=logging.INFO)

    if args.tasks_json:
        tasks = load_tasks(args.tasks_json)
    elif args.tasks:
        tasks = load_tasks(args.tasks)
    else:
        raise ValueError("Requires tasks via --tasks or --tasks-json")

    await run(config, tasks)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
