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
    from hud.rl.config import Config

def get_optimizer(config: Config, model: nn.Module) -> Optimizer:
    if config.training.use_8bit_optimizer and BNB_AVAILABLE:
        console.info_log("Using 8-bit AdamW optimizer from bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.training.lr,
            betas=config.training.adam_betas,
            eps=config.training.adam_eps,
        )
    else:
        console.info_log("Using standard FP32 AdamW optimizer")
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.lr,
            betas=config.training.adam_betas,
            eps=config.training.adam_eps,
        )
    return optimizer