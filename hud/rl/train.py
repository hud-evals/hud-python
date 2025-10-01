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
from hud.rl.loss import compute_grpo_loss, sanity_check
from hud.rl.model import get_model
from hud.rl.optimizer import get_optimizer
from hud.rl.parallel_dims import ParallelDims
from hud.rl.parallelize_qwen_2_5 import parallelize_qwen
from hud.rl.distributed import setup_distributed, synchronize, cleanup_distributed, get_world_size
from hud.rl.utils import (
    get_gpu_utilization,
    get_memory_usage,
)
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
        self.ref_model = copy.deepcopy(self.model)

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

        if self.config.dp_replicate > 1 and not self.parallel_dims.fsdp_enabled:
            self.ref_model.to("cuda")

        with console.progress("Computing ref logprobs for batch...") as progress, torch.no_grad():
            for microbatch in batch:
                sample = microbatch.to_device(torch.device("cuda"))
                logits = self.ref_model.forward(**sample.inputs).logits
                logits = logits[:,-1,:]
                logits = logits / sample.temperature

                per_token_logprobs = []
                for row_logits, row_input_ids in zip(logits, sample.inputs["input_ids"]):
                    row_logprobs = torch.nn.functional.log_softmax(row_logits, dim=-1)
                    row_logprobs = row_logprobs[row_input_ids]
                    per_token_logprobs.append(row_logprobs)
                per_token_logprobs = torch.stack(per_token_logprobs)
                sample.ref_logprobs = per_token_logprobs

        self.ref_model.to("cpu")

        if self.config.dp_replicate > 1 and not self.parallel_dims.fsdp_enabled:
            self.model.to("cuda")

        with console.progress("Computing old logprobs for batch...") as progress, torch.no_grad():
            for microbatch in batch:
                sample = microbatch.to_device(torch.device("cuda"))
                logits = self.model.forward(**sample.inputs).logits

                per_token_logprobs = []
                for row_logits, row_input_ids in zip(logits, sample.inputs["input_ids"]):
                    row_logprobs = torch.nn.functional.log_softmax(row_logits, dim=-1)
                    row_logprobs = row_logprobs[row_input_ids]
                    per_token_logprobs.append(row_logprobs)
                per_token_logprobs = torch.stack(per_token_logprobs)
                sample.old_logprobs = per_token_logprobs

        self.model.train()

        training_start_time = time.time()

        training_time = time.time() - training_start_time
        gpu_util = get_gpu_utilization()
        gpu_memory = get_memory_usage()
        console.info_log(
            f"Update duration: {training_time:.2f}s"
        )
        console.info_log(f"GPU Util: {gpu_util:.1f}% | Memory: {gpu_memory:.2f} GB")

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
