"""Advantage calculation utilities for GRPO training."""

import numpy as np
import torch
from typing import TYPE_CHECKING

from hud.rl.logger import console
from hud.rl.types import TrainingSample

if TYPE_CHECKING:
    from hud.types import Trace
    from hud.rl.config import Config


def calculate_advantages(traces: list[Trace], config: Config) -> list[TrainingSample]:
    """Calculate advantages for a group of traces.

    Args:
        traces: List of traces to process
        config: Training configuration

    Returns:
        List of TrainingSample objects with computed advantages
    """
    group_size = config.training.group_size
    batch_level = config.training.batch_level

    if batch_level == "group":
        groups = [traces[i : i + group_size] for i in range(0, len(traces), group_size)]
    elif batch_level == "batch":
        groups = [traces]
    else:
        raise ValueError(f"Invalid batch level: {batch_level}")

    all_samples = []
    for i, group in enumerate(groups):
        rewards = np.array([trace.reward for trace in group])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Calculate advantages
        samples = [TrainingSample(**trace.model_dump()) for trace in group]
        for sample, reward in zip(samples, rewards, strict=True):
            if sample.isError:
                sample.advantage = torch.tensor([0.0])
                continue

            # No std (non-baseline GRPO)
            if config.training.no_std:
                advantage_value = reward - mean_reward
            else:
                # Avoid division by zero
                if std_reward < 1e-6:
                    advantage_value = 0.0
                else:
                    advantage_value = (reward - mean_reward) / std_reward

            # Leave one out RLOO/LOOP
            if config.training.leave_one_out:
                advantage_value = advantage_value * len(group) / (len(group) - 1)

            sample.advantage = torch.tensor([advantage_value])

        console.info_log(
            f"Advantages for group {i} [{mean_reward:.4f} ± {std_reward:.4f}]: "
            f"{[round(sample.advantage.item(), 4) for sample in samples if sample.advantage is not None]}"
        )

        all_samples.extend(samples)

    return all_samples
