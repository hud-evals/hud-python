import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.replicate import replicate
from transformers import (
    AutoConfig,
    ProcessorMixin,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)

try:
    import liger_kernel.transformers
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

from hud.rl.logger import console
from hud.rl.config import ModelConfig, ProcessorConfig

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
        trust_remote_code=config.trust_remote_code,
    )
    model_cfg.use_cache = False

    if config.use_liger and LIGER_AVAILABLE:
        model_type = model_cfg.model_type
        patch_func_name = f"apply_liger_kernel_to_{model_type}"
        patch_func = getattr(liger_kernel.transformers, patch_func_name, None)
        
        if callable(patch_func):
            patch_func()
        else:
            console.warning(f"No Liger patch function found for {model_type}")
    elif config.use_liger and not LIGER_AVAILABLE:
        console.warning("Liger kernel requested but not available.")

    if "vision_config" in model_cfg:
        model = AutoModelForImageTextToText.from_pretrained(config.base_model, config=model_cfg, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=config.trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.base_model, config=model_cfg, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=config.trust_remote_code)

    if config.freeze_vision_tower:
        freeze_vision_tower(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.info_log(f"Trainable params: {trainable_params/1e6:.1f}M / Total params: {total_params/1e6:.1f}M")

    return model

def get_processor(base_model: str, config: ProcessorConfig) -> ProcessorMixin:
    try:
        processor = AutoProcessor.from_pretrained(
            base_model, 
            **config.model_dump()
        )
        return processor
    except ValueError:
        console.info_log("Processor not available, falling back to tokenizer")
        return AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        console.warning(f"Failed to load processor: {e}")
        raise e

def apply_ddp(model: nn.Module, dp_mesh: DeviceMesh) -> None:
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

if __name__ == "__main__":
    model = get_model(ModelConfig(base_model="Qwen/Qwen2.5-VL-3B-Instruct"))
    config = ProcessorConfig(min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)

    processor = get_processor("Qwen/Qwen2.5-VL-3B-Instruct", config)
    _ = model, processor
