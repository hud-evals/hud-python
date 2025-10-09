from pathlib import Path
import json
from typing import Dict
import os
import torch
import torch.distributed as dist


def get_weights_path(output_dir: str | Path, step: int) -> Path:
    output_dir = Path(output_dir)
    return output_dir / f"step_{step:05d}" / "checkpoints" / "model.safetensors"


def save_step_metrics(output_dir: str | Path, step: int, metrics: Dict[str, float]) -> Path:
    """Writes metrics to a JSON file: `<output_dir>/step_{step}/metrics.json`.
    """
    step_dir = Path(output_dir) / f"step_{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(dict(metrics), f)
    return path


# Distributed helpers (migrated from distributed.py)

def setup_distributed() -> None:
    """Initialize torch.distributed (NCCL) and set CUDA device from LOCAL_RANK."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=(torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None),
        )


def get_world_size() -> int:
    """Return expected world size from env, defaulting to 1."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
