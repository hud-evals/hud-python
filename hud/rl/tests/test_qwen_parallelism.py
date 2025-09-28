"""
Test suite for Qwen model parallelism strategies.
Tests DDP, FSDP, and HSDP with different configurations for both LM and VL models.
"""

import os
import torch
import torch.distributed as dist
import pytest
from PIL import Image

from hud.rl.parallel_dims import ParallelDims
from hud.rl.parallelize_qwen_2_5 import parallelize_qwen
from hud.rl.model import get_model, get_processor
from hud.rl.config import ModelConfig
from hud.rl.logger import console
from torch.distributed.fsdp import FSDPModule

console.set_verbose(True)

def _lm_model_default():
    return os.environ.get("QWEN_LM_MODEL", "Qwen/Qwen2.5-0.5B")


def _vl_model_default():
    return os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def setup_environment():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for these parallel tests')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", torch.cuda.current_device()))

    return world_size


def cleanup():
    if dist.is_initialized():
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if current_device is not None:
            dist.barrier(device_ids=[current_device])
        else:
            dist.barrier()
        dist.destroy_process_group()

def generate_lm_dummy_input(batch_size=2, seq_len=128, vocab_size=151936):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {"input_ids": input_ids}


def generate_vl_dummy_input(
    processor,
    batch_size: int = 2,
    seq_len_unused: int = 64,
    vocab_size_unused: int = 151936,
    img_size: int = 224,
):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    text_prompts = [f"{image_token} dummy prompt"] * batch_size

    images = [Image.new("RGB", (img_size, img_size), color=(i * 13 % 255, i * 29 % 255, i * 41 % 255)) for i in range(batch_size)]

    processed = processor(
        images=images,
        text=text_prompts,
        return_tensors="pt",
        padding=True,
    )

    pixel_values = processed["pixel_values"].to(dtype=torch.bfloat16)
    image_grid_thw = processed.get("image_grid_thw")
    if image_grid_thw is None:
        raise ValueError("Processor did not return image_grid_thw")

    input_ids = processed["input_ids"]
    attention_mask = processed.get("attention_mask")

    result = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    if attention_mask is not None:
        result["attention_mask"] = attention_mask
    mm_token_type_ids = processed.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        result["mm_token_type_ids"] = mm_token_type_ids

    return result


def run_parallelism_test(model_name, parallel_dims, test_type, is_vl=False):

    console.info_log(
        f"Starting {test_type} test for {'VL' if is_vl else 'LM'} model: {model_name}"
    )

    config = ModelConfig(base_model=model_name)

    if is_vl:
        config.freeze_vision_tower = True

    model = get_model(config)
    model = parallelize_qwen(model, parallel_dims)

    print(model)

    if test_type in ("FSDP", "HSDP"):
        assert isinstance(model, FSDPModule)

    model.train()

    if parallel_dims.dp_replicate_enabled and not parallel_dims.fsdp_enabled:
      model = model.to("cuda")

    if is_vl:
        try:
            processor = get_processor(config)
        except OSError as exc:
            pytest.skip(f"Processor unavailable: {exc}")
        inputs = generate_vl_dummy_input(processor)
    else:
        inputs = generate_lm_dummy_input()

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Forward pass test
    console.info_log(f"Running forward pass...")
    model.zero_grad(set_to_none=True)
    outputs = model(return_dict=True, **inputs)

    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        logits = outputs[0]
    if logits is None:
        raise AssertionError("Model output did not contain logits")

    model_type = "VL" if is_vl else "LM"
    loss = logits.float().mean()
    loss_value = loss.item()
    console.info_log(f"Running backward pass...")
    loss.backward()

    has_grad = any(
        param.grad is not None
        for param in model.parameters()
        if param.requires_grad
    )
    if not has_grad:
        console.warning(f"No gradients found; likely testing with meta weights")

    grad_norm = 0.0
    grads_with_values = 0
    total_params = 0
    for param in model.parameters():
        total_params += 1
        if param.grad is not None:
            grad_norm += param.grad.data.float().pow(2).sum().cpu().item()
            grads_with_values += 1
    grad_norm = grad_norm ** 0.5

    console.info_log(f"[{model_type} {test_type} backward grad norm: {grad_norm:.6e}")
    console.info_log(f"[{model_type} {test_type} test passed - Loss: {loss_value:.4f}")
    model.zero_grad(set_to_none=True)


# DDP Tests
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_lm_ddp_4_gpus():
    """Test LM DDP with 4 GPUs"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=world_size,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_lm_model_default(), parallel_dims, "DDP", is_vl=False)

    finally:
        cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_vl_ddp_4_gpus():
    """Test VL DDP with 4 GPUs"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=world_size,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_vl_model_default(), parallel_dims, "DDP", is_vl=True)

    finally:
        cleanup()


# FSDP Tests
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_lm_fsdp_4_gpus():
    """Test LM FSDP with 4 GPUs"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_lm_model_default(), parallel_dims, "FSDP", is_vl=False)

    finally:
        cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_vl_fsdp_4_gpus():
    """Test VL FSDP with 4 GPUs"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_vl_model_default(), parallel_dims, "FSDP", is_vl=True)

    finally:
        cleanup()


# HSDP Tests
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_lm_hsdp_2x2_mesh():
    """Test LM HSDP with 2x2 mesh"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_lm_model_default(), parallel_dims, "HSDP", is_vl=False)

    finally:
        cleanup()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_vl_hsdp_2x2_mesh():
    """Test VL HSDP with 2x2 mesh"""
    world_size = setup_environment()
    if world_size < 4:
        pytest.skip(f"Requires world_size >= 4, got {world_size}")

    try:
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size
        )

        run_parallelism_test(_vl_model_default(), parallel_dims, "HSDP", is_vl=True)

    finally:
        cleanup()





if __name__ == "__main__":
    """Run tests manually with torchrun"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: torchrun --nproc_per_node=4 test_qwen_parallelism.py <test_name>")
        print("Available tests:")
        print("  LM: lm_ddp, lm_fsdp, lm_hsdp")
        print("  VL: vl_ddp, vl_fsdp, vl_hsdp")
        print("  All LM: all_lm")
        print("  All VL: all_vl")
        print("  All: all")
        sys.exit(1)

    test_name = sys.argv[1]

    # LM tests
    if test_name == "lm_ddp":
        test_lm_ddp_4_gpus()
    elif test_name == "lm_fsdp":
        test_lm_fsdp_4_gpus()
    elif test_name == "lm_hsdp":
        test_lm_hsdp_2x2_mesh()

    # VL tests
    elif test_name == "vl_ddp":
        test_vl_ddp_4_gpus()
    elif test_name == "vl_fsdp":
        test_vl_fsdp_4_gpus()
    elif test_name == "vl_hsdp":
        test_vl_hsdp_2x2_mesh()

    # Combined tests
    elif test_name == "all_lm":
        print("Running all LM tests...")
        test_lm_ddp_4_gpus()
        test_lm_fsdp_4_gpus()
        test_lm_hsdp_2x2_mesh()
    elif test_name == "all_vl":
        print("Running all VL tests...")
        test_vl_ddp_4_gpus()
        test_vl_fsdp_4_gpus()
        test_vl_hsdp_2x2_mesh()
    elif test_name == "all":
        print("Running all tests...")
        test_lm_ddp_4_gpus()
        test_lm_fsdp_4_gpus()
        test_lm_hsdp_2x2_mesh()
        test_vl_ddp_4_gpus()
        test_vl_fsdp_4_gpus()
        test_vl_hsdp_2x2_mesh()
    else:
        print(f"Unknown test: {test_name}")
        print("Available tests:")
        print("  LM: lm_ddp, lm_fsdp, lm_hsdp")
        print("  VL: vl_ddp, vl_fsdp, vl_hsdp")
        print("  All LM: all_lm")
        print("  All VL: all_vl")
        print("  All: all")
        sys.exit(1)
