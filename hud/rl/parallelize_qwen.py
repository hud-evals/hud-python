import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy, fully_shard
from transformers import Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration

from hud.rl.model import apply_ddp
from hud.rl.parallel_dims import ParallelDims
from hud.rl.logger import console

def apply_fsdp_lm(
    model: Qwen2ForCausalLM,
    dp_mesh: DeviceMesh,
    cpu_offload: bool,
    reshard_after_forward: bool,
) -> None:
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_config = {
        "mesh": dp_mesh, "mp_policy": policy
    }
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffloadPolicy()

    for block in model.model.layers:
        fully_shard(block, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)

    if hasattr(model, "config") and not model.config.tie_word_embeddings:
        fully_shard(model.model.embed_tokens, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
        fully_shard([model.lm_head, model.model.norm], mesh=dp_mesh, mp_policy=policy, reshard_after_forward=False)
    
    console.info_log("Applied FSDP to the model")

def apply_fsdp_vl(
    model: Qwen2_5_VLForConditionalGeneration,
    dp_mesh: DeviceMesh,
    cpu_offload: bool,
    reshard_after_forward: bool,
) -> None:
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_config = {
        "mesh": dp_mesh, "mp_policy": policy
    }
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffloadPolicy()

    fully_shard([model.model.visual.patch_embed, model.model.visual.rotary_pos_emb], mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)

    for block in model.model.visual.blocks:
        fully_shard(block, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    
    for block in model.model.language_model.layers:
        fully_shard(block, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)

    if hasattr(model, "config") and not model.config.tie_word_embeddings:
        fully_shard(model.model.language_model.embed_tokens, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
        fully_shard([model.lm_head, model.model.language_model.norm], mesh=dp_mesh, mp_policy=policy, reshard_after_forward=False)

    console.info_log("Applied FSDP to the model")

def parallelize_qwen(
    model: Qwen2ForCausalLM | Qwen2_5_VLForConditionalGeneration,
    parallel_dims: ParallelDims,
):
    """
    Apply data parallelism (DDP/FSDP/HSDP) to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.world_mesh

    if parallel_dims.fsdp_enabled:
        # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate")

        if isinstance(model, Qwen2ForCausalLM):
            apply_fsdp_lm(model, world_mesh[tuple(dp_mesh_dim_names)], cpu_offload=False, reshard_after_forward=False)
        elif isinstance(model, Qwen2_5_VLForConditionalGeneration):
            apply_fsdp_vl(model, world_mesh[tuple(dp_mesh_dim_names)], cpu_offload=False, reshard_after_forward=False)

        if parallel_dims.dp_replicate_enabled:
            console.info_log("Applied HSDP to the model")
        else:
            console.info_log("Applied FSDP to the model")

    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
        )

    return model
