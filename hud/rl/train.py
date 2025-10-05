from __future__ import annotations

import argparse
import os
import copy
import time
from typing import Sequence

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from hud.rl.checkpoint import CheckpointManager
from hud.rl.config import TrainingConfig
from hud.rl.logger import console
from hud.rl.loss import compute_loss, get_per_token_logps, entropy_from_logits
from hud.rl.model import get_model
from hud.rl.optimizer import get_optimizer
from hud.rl.parallel_dims import ParallelDims
from hud.rl.parallelize_qwen_2_5 import parallelize_qwen
from hud.rl.distributed import setup_distributed, synchronize, cleanup_distributed, get_world_size
from hud.rl.types import TrainingSample


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

        setup_distributed()
        world_size = get_world_size()
        self.parallel_dims = ParallelDims(
            dp_replicate=self.config.dp_replicate,
            dp_shard=self.config.dp_shard,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )

        self.model = get_model(config.model)
        self.model = parallelize_qwen(self.model, self.parallel_dims)
        if self.config.loss.kl_beta > 0:
            self.ref_model = get_model(config.model)
            self.ref_model = parallelize_qwen(self.ref_model, self.parallel_dims)
            self.ref_model.eval()
        else:
            self.ref_model = None

        self.optimizer = get_optimizer(self.config.optimizer, self.model)

        self.checkpoint_manager = CheckpointManager(
            out_dir=config.checkpoint.out_dir,
            checkpoint_prefix=config.checkpoint.checkpoint_prefix,
        )

        self.step_count = 0

    def get_batch(
        self
    ) -> list[TrainingSample]:
        raise NotImplementedError(
            "Trainer.get_batch is not implemented; pass microbatches explicitly"
        )

    def train_step(
        self,
        batch: list[TrainingSample],
    ) -> None:

        # Compute ref logprobs if we're using KL regularization
        if self.ref_model:
            with console.progress("Computing ref logprobs for batch...") as progress, torch.no_grad():
                for minibatch in batch:
                    sample = minibatch.to_device(torch.device("cuda"))
                    logits = self.ref_model(**sample.inputs).logits
                    logprobs = get_per_token_logps(logits, sample.inputs["input_ids"], sample.temperature)
                    sample.ref_logprobs = logprobs
                    del logits, logprobs


        with console.progress("Computing old logprobs for batch...") as progress, torch.no_grad():
            for minibatch_idx, minibatch in enumerate(batch):
                sample = minibatch.to_device(torch.device("cuda"))
                logits = self.model(**sample.inputs).logits
                logprobs = get_per_token_logps(logits, sample.inputs["input_ids"], sample.temperature)

                sample.old_logprobs = logprobs

        self.model.train()

        training_start_time = time.time()

        if self.config.loss.norm_type == "token":
            # Normalize with the number of assistant tokens in the local batch
            loss_norm = sum(minibatch.inputs["assistant_mask"].sum().item() for minibatch in batch)
        elif self.config.loss.norm_type == "sequence":
            loss_norm = len(batch)

        with console.progress("Training batch...") as progress:
            for step, minibatch in enumerate(batch):
                if hasattr(self.model, 'set_requires_all_reduce'):
                    self.model.set_requires_all_reduce(step == len(batch) - 1)  # type: ignore

                sample = minibatch.to_device(torch.device("cuda"))
                logits = self.model(**sample.inputs).logits
                logprobs = get_per_token_logps(logits, sample.inputs["input_ids"], sample.temperature)

                with torch.no_grad():
                    entropy = entropy_from_logits(logits)

                loss, loss_dict = compute_loss(logprobs, sample.old_logprobs, sample.ref_logprobs, sample.advantage, sample.inputs["assistant_mask"], self.config.loss, loss_norm)

                del logits
                
                loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        training_time = time.time() - training_start_time
        console.info_log(
            f"Update duration: {training_time:.2f}s"
        )

    def cleanup(self) -> None:
        try:
            synchronize()
        except Exception:
            console.warning("Failed to synchronize during cleanup")

        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO trainer")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    _ = parser.parse_args()
    console.error("Standalone trainer execution is not supported; use the runner workflow.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
