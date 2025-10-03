"""vLLM server management utilities."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx
from rich.console import Console

from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)

console = Console()


def get_vllm_args(model_name: str, chat_template_path: Path | None = None) -> list[str]:
    """Get common vLLM server arguments for both local and remote deployments."""
    args = [
        "serve",
        model_name,
        "--api-key",
        "token-abc123",
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--max-model-len",
        "16384",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "4",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--disable-log-requests",
        "--dtype",
        "auto",
    ]

    # Add chat template if provided
    if chat_template_path and chat_template_path.exists():
        args.extend(["--chat-template", str(chat_template_path.absolute())])

    return args


def check_vllm_server() -> bool:
    """Check if vLLM server is running."""
    try:
        response = httpx.get("http://localhost:8000/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def kill_vllm_server() -> None:
    """Kill any running vLLM server processes."""
    try:
        # Check for PID file first
        pid_file = Path("/tmp/vllm_server.pid")  # noqa: S108
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                subprocess.run(["kill", "-TERM", str(pid)], check=False)  # noqa: S603, S607
                time.sleep(2)
                # Force kill if still running
                subprocess.run(["kill", "-9", str(pid)], check=False)  # noqa: S603, S607
                pid_file.unlink()
            except Exception as e:
                hud_console.error(f"Failed to kill vLLM server: {e}")

        # Also try to kill by process name
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)  # noqa: S607
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)  # noqa: S607
        time.sleep(2)

        # Check for any process using port 8000
        result = subprocess.run(["lsof", "-ti:8000"], capture_output=True, text=True, check=False)  # noqa: S607

        if result.stdout.strip():
            for pid in result.stdout.strip().split("\n"):
                try:
                    subprocess.run(["kill", "-9", pid], check=False)  # noqa: S603, S607
                except Exception as e:
                    hud_console.error(f"Failed to kill vLLM server: {e}")

        console.print("[yellow]Killed existing vLLM server processes[/yellow]")
    except Exception as e:
        hud_console.error(f"Error killing vLLM server: {e}")


def start_vllm_server(model_name: str, gpu_index: int = 1, restart: bool = False) -> None:
    """Start vLLM server in the background with dynamic GPU selection."""
    if restart:
        kill_vllm_server()
        time.sleep(3)

    # Check if already running
    if check_vllm_server():
        console.print("[green]vLLM server is already running[/green]")
        return

    console.print(f"[cyan]Starting vLLM server with {model_name} on GPU {gpu_index}...[/cyan]")

    # Set up environment variables
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_LOGGING_LEVEL": "INFO",  # Changed from DEBUG to reduce noise
            "CUDA_LAUNCH_BLOCKING": "1",  # Better error messages
        }
    )

    # Get the path to chat template
    chat_template_path = Path(__file__).parent.parent.parent / "rl" / "chat_template.jinja"

    # Build the vLLM command
    vllm_args = get_vllm_args(model_name, chat_template_path)
    cmd = ["uv", "run", "vllm", *vllm_args]

    # Start the server in the background
    with open("/tmp/vllm_server.log", "w") as log_file:  # noqa: S108,
        process = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,  # type: ignore
            cwd=Path.cwd(),  # Use current working directory
        )

    console.print("[yellow]vLLM server starting in background...[/yellow]")
    console.print(f"[yellow]Process ID: {process.pid}[/yellow]")
    console.print("[yellow]Check logs at: /tmp/vllm_server.log[/yellow]")

    # Save PID for later management
    pid_file = Path("/tmp/vllm_server.pid")  # noqa: S108
    pid_file.write_text(str(process.pid))


async def wait_for_vllm_server(timeout: int = 360) -> bool:  # noqa: ASYNC109
    """Wait for vLLM server to be ready."""
    start_time = time.time()
    console.print("[yellow]Waiting for vLLM server to be ready (up to 6 minutes)...[/yellow]")

    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get("http://localhost:8000/health", timeout=2.0)
                if response.status_code == 200:
                    console.print("[green]✅ vLLM server is ready![/green]")
                    return True
            except httpx.ConnectError:
                pass
            except Exception as e:
                hud_console.error(f"Failed to connect to vLLM server: {e}")

            await asyncio.sleep(2)
            elapsed = int(time.time() - start_time)
            console.print(f"[yellow]Waiting... ({elapsed}s / {timeout}s)[/yellow]", end="\r")

    console.print("\n[red]❌ vLLM server failed to start within timeout[/red]")
    console.print("[yellow]Check /tmp/vllm_server.log for details[/yellow]")
    return False
