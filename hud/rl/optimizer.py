from typing import TYPE_CHECKING
from torch import nn
from torch.optim import AdamW, Optimizer

try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from hud.rl.logger import console

if TYPE_CHECKING:
    from hud.rl.config import OptimizerConfig

def get_optimizer(config: "OptimizerConfig", model: nn.Module) -> Optimizer:
    if config.use_8bit_optimizer and BNB_AVAILABLE:
        console.info_log("Using 8-bit AdamW optimizer from bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )
    else:
        console.info_log("Using standard FP32 AdamW optimizer")
        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )
    return optimizer