"""Test script to visualize shapes during train_step using pre-saved batch."""

import os
import torch

from hud.rl.config import TrainingConfig, ModelConfig
from hud.rl.logger import console
from hud.rl.train import Trainer


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    batch_file = f"/home/ubuntu/hud-python/hud/rl/tests/data/batch_gpu{rank}.pt"

    console.info("=" * 80)
    console.info(f"Loading batch from {batch_file}...")

    batch = torch.load(batch_file, weights_only=False)

    console.info(f"Loaded {len(batch)} minibatch(es) for GPU {rank}")
    console.info("=" * 80)
    console.info("Initializing trainer...")

    training_config = TrainingConfig()
    training_config.model = ModelConfig(base_model="Qwen/Qwen2.5-VL-7B-Instruct")
    training_config.dp_shard = 4
    training_config.optimizer.use_8bit_optimizer = False
    trainer = Trainer(training_config)

    console.info("=" * 80)
    console.info(f"Running train_step on rank {rank}...")

    trainer.train_step(batch)

    console.info("=" * 80)
    console.info("Cleaning up...")
    trainer.cleanup()


if __name__ == "__main__":
    main()
