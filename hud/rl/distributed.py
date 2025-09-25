"""Distributed training utilities for GRPO."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def setup_distributed() -> None:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Set device for this process
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group("nccl")


def get_local_rank() -> int:
    """Get local rank from environment."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank() -> int:
    """Get global rank from environment."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get world size from environment."""
    return int(os.environ.get("WORLD_SIZE", 1))


def cleanup_distributed() -> None:
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all ranks.

    Args:
        obj: Object to broadcast (used on src rank)
        src: Source rank
        device: Device for temporary tensor buffer during pickling transfer
    """
    if not dist.is_initialized():
        return obj

    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def scatter_object(
    obj_list: list[Any] | None,
    src: int = 0,
) -> Any:
    """Scatter a list of Python objects from src so each rank receives one object.

    Usage:
        - On src rank: pass the full list (length == world_size)
        - On non-src ranks: pass None

    Returns:
        The object intended for this rank.
    """
    if not dist.is_initialized():
        # Single-process: return first element if provided, else None
        if obj_list is None or len(obj_list) == 0:
            return None
        return obj_list[0]

    out: list[Any] = [None]
    if dist.get_rank() == src:
        dist.scatter_object_list(out, obj_list, src=src)
    else:
        dist.scatter_object_list(out, None, src=src)
    return out[0]


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor] | None:
    """Gather tensors from all ranks to rank 0.

    Returns:
        List of tensors on rank 0, None on other ranks
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gathered, dst=0)
        return gathered
    else:
        dist.gather(tensor, None, dst=0)
        return None
