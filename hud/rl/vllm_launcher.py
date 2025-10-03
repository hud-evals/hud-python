"""vLLM server management utilities."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

from hud.rl.vllm_server import start_server_with_multiprocessing
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)

console = Console()


class VLLMServerManager:
    """Manager class for vLLM server processes using multiprocessing."""
    
    def __init__(self):
        self._process: Optional[multiprocessing.Process] = None
        self._pid_file = Path("/tmp/vllm_server.pid")  # noqa: S108
        self._log_file = Path("/tmp/vllm_server.log")  # noqa: S108
    
    @property
    def is_running(self) -> bool:
        """Check if the server process is currently running."""
        if self._process and self._process.is_alive():
            return True
        # Also check if server is responding on the port
        return check_vllm_server()
    
    @property
    def pid(self) -> Optional[int]:
        """Get the PID of the running server process."""
        if self._process:
            return self._process.pid
        # Try to read from PID file as fallback
        if self._pid_file.exists():
            try:
                return int(self._pid_file.read_text().strip())
            except Exception:
                pass
        return None
    
    def start(self, model_name: str, gpu_index: int = 1, restart: bool = False) -> None:
        """Start vLLM server in the background with dynamic GPU selection using multiprocessing."""
        if restart:
            self.kill()
            time.sleep(3)
        
        # Check if already running
        if self.is_running:
            console.print("[green]vLLM server is already running[/green]")
            return
        
        console.print(f"[cyan]Starting vLLM server with {model_name} on GPU {gpu_index}...[/cyan]")
        
        # Set up environment variables
        os.environ.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                "TOKENIZERS_PARALLELISM": "false",
                "VLLM_LOGGING_LEVEL": "INFO",
                "CUDA_LAUNCH_BLOCKING": "1",
            }
        )
        
        # Get the path to chat template
        chat_template_path = Path(__file__).parent.parent.parent / "rl" / "chat_template.jinja"
        
        # Build the vLLM arguments
        vllm_args = get_vllm_args(model_name, chat_template_path)
        
        # Start the server using multiprocessing
        self._process = start_server_with_multiprocessing(
            vllm_args=vllm_args,
            daemon=False,  # Don't use daemon so we can properly manage the process
            name="vLLMServer"
        )
        
        console.print("[yellow]vLLM server starting in background...[/yellow]")
        console.print(f"[yellow]Process ID: {self._process.pid}[/yellow]")
        console.print(f"[yellow]Log file: {self._log_file}[/yellow]")
        
        # Save PID for later management
        self._pid_file.write_text(str(self._process.pid))
    
    def kill(self) -> None:
        """Kill any running vLLM server processes."""
        try:
            # First try to terminate the multiprocessing process if it exists
            if self._process and self._process.is_alive():
                console.print("[yellow]Terminating vLLM server process...[/yellow]")
                self._process.terminate()
                self._process.join(timeout=5)  # Wait up to 5 seconds
                
                if self._process.is_alive():
                    console.print("[yellow]Force killing vLLM server process...[/yellow]")
                    self._process.kill()
                    self._process.join(timeout=2)
                
                self._process = None
            
            # Clean up PID file and kill by PID as fallback
            if self._pid_file.exists():
                try:
                    pid = int(self._pid_file.read_text().strip())
                    subprocess.run(["kill", "-TERM", str(pid)], check=False)  # noqa: S603, S607
                    time.sleep(2)
                    # Force kill if still running
                    subprocess.run(["kill", "-9", str(pid)], check=False)  # noqa: S603, S607
                    self._pid_file.unlink()
                except Exception as e:
                    hud_console.error(f"Failed to kill vLLM server by PID: {e}")
            
            # Also try to kill by process name as final fallback
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
                        hud_console.error(f"Failed to kill process on port 8000: {e}")
            
            console.print("[yellow]Killed existing vLLM server processes[/yellow]")
        except Exception as e:
            hud_console.error(f"Error killing vLLM server: {e}")
    
    def wait_until_ready(self, timeout: int = 360) -> bool:
        """
        Wait for the vLLM server to be ready (synchronous version).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False if timeout reached
        """
        start_time = time.time()
        console.print(f"[yellow]Waiting for vLLM server to be ready (up to {timeout} seconds)...[/yellow]")
        
        while time.time() - start_time < timeout:
            if check_vllm_server():
                console.print("[green]✅ vLLM server is ready![/green]")
                return True
            
            time.sleep(2)
            elapsed = int(time.time() - start_time)
            console.print(f"[yellow]Waiting... ({elapsed}s / {timeout}s)[/yellow]", end="\r")
        
        console.print(f"\n[red]❌ vLLM server failed to start within {timeout} seconds[/red]")
        console.print(f"[yellow]Check {self._log_file} for details[/yellow]")
        return False
    
    async def wait_until_ready_async(self, timeout: int = 360) -> bool:
        """
        Wait for the vLLM server to be ready (async version).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False if timeout reached
        """
        return await wait_for_vllm_server(timeout)


# Create a singleton instance for backward compatibility
_server_manager = VLLMServerManager()


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
    """Kill any running vLLM server processes.
    
    This is a convenience wrapper around VLLMServerManager for backward compatibility.
    """
    _server_manager.kill()


# Backward compatibility functions that delegate to the manager
def start_vllm_server(model_name: str, gpu_index: int = 1, restart: bool = False) -> None:
    """Start vLLM server in the background with dynamic GPU selection.
    
    This is a convenience wrapper around VLLMServerManager for backward compatibility.
    """
    _server_manager.start(model_name, gpu_index, restart)


async def wait_for_vllm_server(timeout: int = 360) -> bool:  # noqa: ASYNC109
    """Wait for vLLM server to be ready.
    
    This is a convenience wrapper around VLLMServerManager for backward compatibility.
    """
    start_time = time.time()
    console.print(f"[yellow]Waiting for vLLM server to be ready (up to {timeout//60} minutes)...[/yellow]")

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

    console.print(f"\n[red]❌ vLLM server failed to start within {timeout} seconds[/red]")
    console.print(f"[yellow]Check {_server_manager._log_file} for details[/yellow]")
    return False


if __name__ == "__main__":
    """Run the vLLM server with sensible defaults when script is executed directly."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Action to perform on the vLLM server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name to serve (default: NousResearch/Hermes-2-Pro-Llama-3-8B)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU index to use (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=360,
        help="Timeout in seconds for server startup (default: 360)"
    )
    
    args = parser.parse_args()
    
    # Create server manager instance
    manager = VLLMServerManager()
    
    try:
        if args.action == "start":
            console.print(f"[cyan]Starting vLLM server with model: {args.model}[/cyan]")
            manager.start(args.model, gpu_index=args.gpu)
            
            # Wait for server to be ready
            if manager.wait_until_ready(timeout=args.timeout):
                console.print("[green]✨ vLLM server is running successfully![/green]")
                console.print("[cyan]Server URL: http://localhost:8000[/cyan]")
                console.print("[cyan]API Key: token-abc123[/cyan]")
                sys.exit(0)
            else:
                console.print("[red]Failed to start vLLM server[/red]")
                sys.exit(1)
                
        elif args.action == "stop":
            console.print("[yellow]Stopping vLLM server...[/yellow]")
            manager.kill()
            console.print("[green]✅ vLLM server stopped[/green]")
            
        elif args.action == "restart":
            console.print("[yellow]Restarting vLLM server...[/yellow]")
            manager.start(args.model, gpu_index=args.gpu, restart=True)
            
            if manager.wait_until_ready(timeout=args.timeout):
                console.print("[green]✨ vLLM server restarted successfully![/green]")
                console.print("[cyan]Server URL: http://localhost:8000[/cyan]")
                console.print("[cyan]API Key: token-abc123[/cyan]")
                sys.exit(0)
            else:
                console.print("[red]Failed to restart vLLM server[/red]")
                sys.exit(1)
                
        elif args.action == "status":
            if manager.is_running:
                console.print("[green]✅ vLLM server is running[/green]")
                if manager.pid:
                    console.print(f"[cyan]Process ID: {manager.pid}[/cyan]")
                console.print("[cyan]Server URL: http://localhost:8000[/cyan]")
            else:
                console.print("[yellow]⚠️  vLLM server is not running[/yellow]")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Unexpected error in vLLM launcher")
        sys.exit(1)
