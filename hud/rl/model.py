import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModel, ProcessorMixin, AutoTokenizer, AutoProcessor

try:
    import liger_kernel.transformers
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

from hud.rl.distributed import get_local_rank, get_world_size
from hud.rl.logger import console
from hud.rl.config import ModelConfig

def freeze_vision_tower(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ["visual", "vision"]):
            for p in module.parameters():
                p.requires_grad_(False)
            module.eval()

def get_model(config: ModelConfig) -> nn.Module:
    console.info_log(f"Loading model: {config.base_model}")
    console.info_log(f"Using {config.attn_implementation} attn implementation")

    model_cfg = AutoConfig.from_pretrained(
        config.base_model, 
        attn_implementation=config.attn_implementation, 
        trust_remote_code=config.trust_remote_code
    )
    model_cfg.use_cache = False

    if config.use_liger and LIGER_AVAILABLE:
        console.info_log("Applying Liger kernel")
        model_type = model_cfg.model_type
        patch_func_name = f"apply_liger_kernel_to_{model_type}"
        patch_func = getattr(liger_kernel.transformers, patch_func_name, None)
        
        if callable(patch_func):
            patch_func()
            console.info_log(f"Applied Liger-Kernel patch to {model_type}")
        else:
            console.warning(f"No Liger patch function found for {model_type}")
    elif config.use_liger and not LIGER_AVAILABLE:
        console.warning("Liger kernel requested but not available.")
    
    model = AutoModel.from_pretrained(config.base_model, config=model_cfg, dtype=torch.bfloat16)

    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
    )

    model = get_peft_model(model, lora_config)

    if config.freeze_vision_tower:
        freeze_vision_tower(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.info_log(f"Trainable params: {trainable_params/1e6:.1f}M / Total params: {total_params/1e6:.1f}M")

    world_size = get_world_size()
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        console.info_log("Wrapped model with DDP")

    return model

def get_processor(config: ModelConfig) -> ProcessorMixin:
    try:
        processor = AutoProcessor.from_pretrained(
            config.base_model, 
            trust_remote_code=config.trust_remote_code,
            **config.processor.model_dump()
        )
        return processor
    except ValueError:
        console.info_log("Processor not available, falling back to tokenizer")
        return AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=config.trust_remote_code)
    except Exception as e:
        console.warning(f"Failed to load processor: {e}")
        raise e

if __name__ == "__main__":
    config = ModelConfig(base_model="Qwen/Qwen2.5-3B-Instruct")

    model = get_model(config)
    processor = get_processor(config)
    model.print_trainable_parameters() # type: ignore
