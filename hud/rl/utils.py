from pathlib import Path
import json
import subprocess
from typing import Dict
import os
import torch
import torch.distributed as dist
from hud.rl.logger import console


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

# source: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py#L68
# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, AMD MI325X and Intel PVC
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        console.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 2.25e15
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        console.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12

