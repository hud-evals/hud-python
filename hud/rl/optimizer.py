from typing import TYPE_CHECKING, Any

import torch

try:
    import bitsandbytes as bnb  # type: ignore

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from hud.rl.logger import console

if TYPE_CHECKING:
    from .config import Config


def create_optimizer(config: Config, policy: Any) -> Any:
    """Create optimizer for the policy model.

    Args:
        config: Training configuration
        policy: The policy model (may be wrapped with DDP)

    Returns:
        Configured optimizer
    """
    # Create optimizer - need to access underlying model if DDP
    base_model = policy.module if hasattr(policy, "module") else policy
    trainable_params = [p for _, p in base_model.named_parameters() if p.requires_grad]  # type: ignore

    # Use 8-bit optimizer if configured
    if config.training.use_8bit_optimizer and BNB_AVAILABLE:
        console.info_log("Using 8-bit AdamW optimizer from bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=config.training.lr,
            betas=config.training.adam_betas,
            eps=config.training.adam_eps,
        )
    else:
        console.info_log("Using standard FP32 AdamW optimizer")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.lr,
            betas=config.training.adam_betas,
            eps=config.training.adam_eps,
        )

    # Log optimizer info
    console.info_log(f"Optimizer: {type(optimizer).__name__}")
    num_params = sum(p.numel() for p in trainable_params)
    console.info_log(f"Number of trainable parameters: {num_params:,}")

    return optimizer
