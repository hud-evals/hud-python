from typing import Literal

import torch
from hud.rl.logger import console

def calculate_advantages(
    rewards: torch.Tensor,
    group_size: int = 1,
    scale_rewards: Literal["group", "batch", "none"] = "group",
    leave_one_out: bool = False,
) -> torch.Tensor:
    """
    Calculate advantages for a batch of rewards.

    Args:
        rewards: Tensor of rewards
        group_size: Number of rewards per group (ignored if scale_rewards="batch")
        scale_rewards: How to scale rewards:
            - "group": z-score normalization within each group (standard GRPO)
            - "batch": z-score normalization across entire batch (ignores grouping, LitePPO)
            - "none": no normalization, just subtract mean (Dr.GRPO)
        leave_one_out: Apply RLOO scaling factor G/(G-1) (only used with scale_rewards="none")

    Returns:
        Tensor of advantages
    """
    if scale_rewards == "batch":
        mean = rewards.mean()
        std = rewards.std(unbiased=False)

        if std < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean) / std

        console.info_log(
            f"Batch advantages [{mean.item():.4f} ± {std.item():.4f}]: "
            f"{[round(adv.item(), 4) for adv in advantages]}"
        )
    else:
        num_groups = len(rewards) // group_size
        grouped_rewards = rewards[:num_groups * group_size].view(num_groups, group_size)
        group_means = grouped_rewards.mean(dim=1, keepdim=True)  # [num_groups, 1]
        group_stds = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)  # [num_groups, 1]

        if scale_rewards == "none":
            grouped_advantages = grouped_rewards - group_means

            if leave_one_out and group_size > 1:
                grouped_advantages = grouped_advantages * group_size / (group_size - 1)
        else:
            safe_stds = torch.where(group_stds < 1e-6, torch.ones_like(group_stds), group_stds)
            grouped_advantages = torch.where(
                group_stds < 1e-6,
                torch.zeros_like(grouped_rewards),
                (grouped_rewards - group_means) / safe_stds
            )

        advantages = grouped_advantages.view(-1)

        for i in range(num_groups):
            group_idx = i * group_size
            console.info_log(
                f"Group {i} advantages [{group_means[i].item():.4f} ± {group_stds[i].item():.4f}]: "
                f"{[round(advantages[group_idx + j].item(), 4) for j in range(group_size)]}"
            )

    return advantages
