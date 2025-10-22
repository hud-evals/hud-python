"""Script to prepare and save training batches for testing (one per GPU)."""

import json
from pathlib import Path

import torch

from hud.rl.advantages import calculate_advantages
from hud.rl.batch import batch_samples
from hud.rl.config import Config
from hud.rl.model import get_processor
from hud.rl.preprocessor import preprocess_traces
from hud.rl.types import TrainingSample
from hud.types import Trace


def resolve_pad_token_id(processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)


def main():
    config = Config()
    trace_file = "/home/ubuntu/myworkspace/hud-python/hud/rl/tests/data/traces_de8ea147-3c52-4117-ad24-d1dbaa39a088.json"

    print("=" * 80)
    print("Loading traces from dump...")
    with open(trace_file) as f:
        trace_data = json.load(f)

    traces = [Trace.model_validate(t) for t in trace_data]
    print(f"Loaded {len(traces)} traces")

    print("\n" + "=" * 80)
    print("Preparing batch...")

    processor = get_processor(config.base_model, config.processor)
    pad_token_id = resolve_pad_token_id(processor)

    group_size = 8
    num_traces = min(len(traces), 16)
    traces = traces[:num_traces]

    rewards = torch.tensor([float(trace.reward) for trace in traces], dtype=torch.float32)
    print(f"\nRewards: shape={tuple(rewards.shape)}, values={rewards.tolist()}")

    advantages = calculate_advantages(
        rewards=rewards,
        group_size=group_size,
        scale_rewards="group",
        leave_one_out=False,
    )
    print(f"Advantages: shape={tuple(advantages.shape)}, values={advantages.tolist()}")

    temperatures = torch.tensor(
        [trace.info.get("temperature", 1.0) for trace in traces],
        dtype=torch.float32,
    )
    print(f"Temperatures: shape={tuple(temperatures.shape)}")

    print("\nPreprocessing traces...")
    processed_inputs = preprocess_traces(traces, processor)
    print(f"Processed {len(processed_inputs)} inputs")

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

    mini_batch_size = 1
    num_gpus = 4
    training_batch = batch_samples(samples, mini_batch_size, num_gpus, pad_token_id)

    print("\n" + "=" * 80)
    print(f"Training batch structure: {len(training_batch)} GPU(s)")
    total_minibatches = sum(len(gpu_batch) for gpu_batch in training_batch)
    print(f"Total minibatches: {total_minibatches}")

    for gpu_idx, gpu_batches in enumerate(training_batch):
        print(f"\nGPU {gpu_idx}: {len(gpu_batches)} minibatch(es)")
        for batch_idx, minibatch in enumerate(gpu_batches):
            print(f"  Minibatch {batch_idx}:")
            print(f"    input_ids: {tuple(minibatch.inputs['input_ids'].shape)}")
            print(f"    advantage: {tuple(minibatch.advantage.shape)}")

    print("\n" + "=" * 80)
    print("Saving batches...")

    tests_root = Path(__file__).resolve().parents[1]
    outputs_root = tests_root / "outputs"

    for step in range(5):
        step_dir = outputs_root / f"step_{step:05d}" / "rollouts"
        step_dir.mkdir(parents=True, exist_ok=True)

        for gpu_idx, gpu_batch in enumerate(training_batch):
            output_file = step_dir / f"rank_{gpu_idx}.pt"
            torch.save(gpu_batch, output_file)
            print(f"  GPU {gpu_idx}: {output_file}")

        print("Done!")


if __name__ == "__main__":
    main()
