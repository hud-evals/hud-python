import torch
import torch.nn as nn
from typing import Dict, Any
from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy, fully_shard

from hud.rl.parallel_dims import ParallelDims
from hud.rl.logger import console


def _shard_language_stack(model: nn.Module, fsdp_config: Dict[str, Any]) -> None:
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers  # type: ignore[attr-defined]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers  # type: ignore[attr-defined]

    if layers is not None:
        for block in layers:
            fully_shard(block, **fsdp_config)

    if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
        tie = True
    else:
        tie = False

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and not tie:
        fully_shard(model.model.embed_tokens, **fsdp_config)  # type: ignore[attr-defined]

    norm_module = None
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "norm"):
        norm_module = model.model.language_model.norm  # type: ignore[attr-defined]
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        norm_module = model.model.norm  # type: ignore[attr-defined]
    if norm_module is not None:
        fsdp_config_no_reshard = {**fsdp_config, "reshard_after_forward": False}
        fully_shard(norm_module, **fsdp_config_no_reshard)

    if hasattr(model, "lm_head"):
        fsdp_config_no_reshard = {**fsdp_config, "reshard_after_forward": False}
        fully_shard(model.lm_head, **fsdp_config_no_reshard)  # type: ignore[attr-defined]


def _shard_vision_stack(model: nn.Module, fsdp_config: Dict[str, Any]) -> None:
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        return

    if hasattr(visual, "patch_embed") and hasattr(visual, "rotary_pos_emb"):
        fully_shard([visual.patch_embed, visual.rotary_pos_emb], **fsdp_config)

    blocks = getattr(visual, "blocks", None)
    if blocks is not None:
        for block in blocks:
            fully_shard(block, **fsdp_config)


def apply_fsdp(
    model: nn.Module,
    parallel_dims: ParallelDims,
    reshard_after_forward: bool = False,
    cpu_offload: bool = False,
) -> nn.Module:
    world_mesh = parallel_dims.world_mesh

    if parallel_dims.dp_replicate_enabled:
        dp_mesh_dim_names = ("dp_replicate", "dp_shard")
    else:
        dp_mesh_dim_names = ("dp_shard",)

    dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

    # Create FSDP config once
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_config: Dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffloadPolicy()

    _shard_vision_stack(model, fsdp_config)
    _shard_language_stack(model, fsdp_config)

    fully_shard(model, **fsdp_config)

    if parallel_dims.dp_replicate_enabled:
        console.info_log("Applied HSDP to the model")
    else:
        console.info_log("Applied FSDP to the model")

    return model
