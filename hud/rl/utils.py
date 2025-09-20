import os
import random

import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_memory_usage() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage (0-100)."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        import nvidia_ml_py as nvml  # type: ignore

        nvml.nvmlInit()
        device_id = torch.cuda.current_device()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        # Fallback: estimate based on memory usage
        # This is less accurate but works without nvidia-ml-py
        return min(100.0, (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100)
