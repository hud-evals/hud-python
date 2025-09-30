import os
from typing import Any

import torch
import torch.distributed as dist


def setup_distributed() -> None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", torch.cuda.current_device()))

def get_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))

def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def synchronize() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not dist.is_initialized():
        return obj

    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor] | None:
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
