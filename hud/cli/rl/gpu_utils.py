"""GPU utilities for DDP training."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from hud.rl.config import Config
hud_console = HUDConsole(logging.getLogger(__name__))


def get_gpu_memory_info() -> dict[int, dict[str, Any]]:
    """Get memory usage information for all GPUs."""

    gpu_memory = {}
    try:
        # Get memory info for all GPUs
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(", ")
            if len(parts) >= 4:
                gpu_idx = int(parts[0])
                memory_used = float(parts[1])
                memory_total = float(parts[2])
                memory_free = float(parts[3])
                gpu_memory[gpu_idx] = {
                    "used_mb": memory_used,
                    "total_mb": memory_total,
                    "free_mb": memory_free,
                    "used_pct": (memory_used / memory_total) * 100,
                }

        # Get process information per GPU
        for gpu_idx in gpu_memory:  # noqa: PLC0206
            cmd = [
                "nvidia-smi",
                "-i",
                str(gpu_idx),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
                processes = []
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        memory_mb = float(parts[1])
                        processes.append({"pid": pid, "memory_mb": memory_mb})
                gpu_memory[gpu_idx]["processes"] = processes
            except Exception as e:
                hud_console.error(f"Failed to get process info for GPU {gpu_idx}: {e}")
                gpu_memory[gpu_idx]["processes"] = []

    except Exception as e:
        hud_console.error(f"Failed to get GPU memory info {e}")
        return {}

    return gpu_memory


def health_check_gpus(gpu_indices: list[int]) -> dict[str, Any]:
    """Perform health check on specified GPUs including memory status.

    Returns:
        Dict with:
        - healthy_gpus: List of healthy GPU indices
        - unhealthy_gpus: Dict of unhealthy GPU index -> error message
        - all_healthy: Boolean indicating if all GPUs are healthy
        - memory_issues: Boolean indicating if there are memory issues
    """
    import torch
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]🏥 GPU Health Check[/bold cyan]")

    # First get memory info
    memory_info = get_gpu_memory_info()

    healthy_gpus = []
    unhealthy_gpus = {}
    memory_issues = []

    # Create a table for results
    table = Table(title="GPU Health Status")
    table.add_column("GPU", style="cyan")
    table.add_column("Memory Usage", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    for gpu_idx in gpu_indices:
        # Memory info
        mem_str = "Unknown"
        if gpu_idx in memory_info:
            mem = memory_info[gpu_idx]
            used_gb = mem["used_mb"] / 1024
            total_gb = mem["total_mb"] / 1024
            mem_str = f"{used_gb:.1f}/{total_gb:.1f} GB ({mem['used_pct']:.0f}%)"

            # Check for high memory usage
            if mem["used_pct"] > 70:
                memory_issues.append(gpu_idx)
                proc_info = f" ({len(mem['processes'])} processes)" if mem["processes"] else ""
                unhealthy_gpus[gpu_idx] = f"High memory usage{proc_info}"
                table.add_row(
                    f"GPU {gpu_idx}", mem_str, "❌ Unhealthy", f"High memory usage{proc_info}"
                )
                continue

        # If no severe memory issue, do accessibility test
        try:
            # Try to allocate a small tensor on the GPU
            torch.cuda.set_device(gpu_idx)
            device = torch.device(f"cuda:{gpu_idx}")

            # Test basic allocation
            test_tensor = torch.zeros(100, 100, device=device)

            # Test computation
            result = torch.matmul(test_tensor, test_tensor)

            # Force synchronization
            torch.cuda.synchronize(device)

            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()

            healthy_gpus.append(gpu_idx)
            table.add_row(f"GPU {gpu_idx}", mem_str, "✅ Healthy", "Passed all tests")

        except Exception as e:
            error_msg = str(e)
            if "busy or unavailable" in error_msg:
                short_msg = "Device busy or unavailable"
            elif "out of memory" in error_msg:
                short_msg = "Insufficient memory"
            else:
                short_msg = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg

            unhealthy_gpus[gpu_idx] = short_msg
            table.add_row(f"GPU {gpu_idx}", mem_str, "❌ Unhealthy", short_msg)

        # Small delay between GPU checks
        time.sleep(0.1)

    console.print(table)

    return {
        "healthy_gpus": healthy_gpus,
        "unhealthy_gpus": unhealthy_gpus,
        "all_healthy": len(unhealthy_gpus) == 0,
        "memory_issues": memory_issues,
    }


def calculate_optimal_gpu_allocation(gpu_info: dict[str, Any], config: Config) -> dict[str, Any]:
    """Calculate optimal GPU allocation for DDP GRPO training.

    Key insight: In GRPO, we want to process groups in parallel.
    Optimal case: num_gpus = num_groups (each GPU processes 1 group).
    """
    devices = gpu_info["devices"]
    available_gpus = [device["index"] for device in devices]

    # Need at least 2 GPUs (1 for training, 1 for vLLM)
    if len(available_gpus) < 2:
        return {"use_ddp": False, "reason": "Need at least 2 GPUs"}

    # Reserve last GPU for vLLM
    vllm_gpu = available_gpus[-1]
    training_gpus = available_gpus[:-1]

    # Calculate number of groups
    batch_size = config.training.batch_size
    group_size = config.training.group_size
    num_groups = batch_size // group_size

    if num_groups == 0:
        num_groups = 1

    # Optimal: Use exactly num_groups GPUs (each processes 1 group in parallel)
    # But cap at available training GPUs
    optimal_gpu_count = min(len(training_gpus), num_groups)

    # Only use DDP if we have more than 1 group and more than 1 GPU
    use_ddp = optimal_gpu_count > 1 and num_groups > 1

    if not use_ddp:
        # Single GPU training
        return {
            "use_ddp": False,
            "reason": f"Single GPU sufficient for {num_groups} group(s)",
            "training_gpus": [training_gpus[0]],
            "vllm_gpu": vllm_gpu,
            "num_groups": num_groups,
        }

    # Use optimal number of GPUs for DDP
    training_gpus = training_gpus[:optimal_gpu_count]

    return {
        "use_ddp": True,
        "training_gpus": training_gpus,
        "vllm_gpu": vllm_gpu,
        "num_groups": num_groups,
        "groups_per_gpu": num_groups / len(training_gpus),
        "parallel_efficiency": min(
            1.0, num_groups / len(training_gpus)
        ),  # 1.0 = perfect load balance
    }


def adjust_config_for_ddp(config: Config, num_gpus: int) -> Config:
    """Adjust configuration for optimal DDP performance.

    Scaling rule:
    - For 1 GPU: batch_size = 2 * group_size
    - For N GPUs (N > 1): batch_size = N * group_size

    This ensures each GPU processes exactly 1 group in parallel for optimal performance.
    """
    group_size = config.training.group_size

    # Apply scaling rule
    if num_gpus == 1:
        # Special case: 2 groups for single GPU
        groups_per_gpu = 2
        config.training.batch_size = 2 * group_size
    else:
        groups_per_gpu = config.training.batch_size // group_size
        # Multi-GPU: each GPU processes groups_per_gpu groups
        config.training.batch_size = num_gpus * group_size * groups_per_gpu

    # Update max_parallel_episodes to match
    config.actor.max_parallel_episodes = config.training.batch_size

    config.training.num_gpus = num_gpus

    # Log the adjustment
    from rich.console import Console

    console = Console()
    console.print(
        f"\n[cyan]📊 Adjusted batch_size to {config.training.batch_size} ({config.training.batch_size // group_size} groups)[/cyan]"  # noqa: E501
    )
    console.print(
        f"[cyan]   Each of the {num_gpus} GPU(s) will process {groups_per_gpu} group(s) in parallel[/cyan]"  # noqa: E501
    )

    return config


def kill_high_memory_processes(memory_threshold: float = 70.0) -> int:
    """Kill all GPU processes using more than threshold% memory.

    Returns:
        Number of processes killed
    """
    from rich.console import Console

    console = Console()

    memory_info = get_gpu_memory_info()
    killed_count = 0

    for gpu_idx, info in memory_info.items():
        if info["used_pct"] > memory_threshold:
            for proc in info.get("processes", []):
                pid = proc["pid"]
                try:
                    # Try graceful termination first
                    subprocess.run(["kill", "-TERM", str(pid)], check=False, capture_output=True)  # noqa: S603, S607
                    killed_count += 1
                    console.print(
                        f"[yellow]Terminating PID {pid} on GPU {gpu_idx} ({proc['memory_mb'] / 1024:.1f} GB)[/yellow]"  # noqa: E501
                    )
                except Exception as e:
                    console.print(f"[red]Failed to kill PID {pid}: {e}[/red]")

    if killed_count > 0:
        console.print(f"\n[yellow]Sent termination signal to {killed_count} processes...[/yellow]")
        time.sleep(3)

        # Force kill any remaining
        for info in memory_info.values():
            for proc in info.get("processes", []):
                pid = proc["pid"]
                try:
                    # Check if still running
                    subprocess.run(  # noqa: S603
                        ["kill", "-0", str(pid)],  # noqa: S607
                        check=True,
                        capture_output=True,
                    )
                    # If no error, process is still running, force kill
                    subprocess.run(["kill", "-KILL", str(pid)], check=False)  # noqa: S603, S607
                    console.print(f"[red]Force killed PID {pid}[/red]")
                except Exception:
                    hud_console.error(f"Failed to kill PID {pid}")

    return killed_count
