#!/usr/bin/env python3
"""Create a local browser provider that uses local Playwright instead of remote services."""

import json
import os
import tempfile

def create_local_kernel_provider():
    """Create a local implementation of the kernel provider."""

    # Path to the kernel provider file
    kernel_provider_path = "environments/remote_browser/src/hud_controller/providers/kernel.py"

    # Read the current kernel provider
    with open(kernel_provider_path, 'r') as f:
        content = f.read()

    # Replace with local implementation
    new_content = '''"""Local browser provider implementation."""

import asyncio
import os
import subprocess
from typing import Any, Dict
from .base import BrowserProvider


class KernelProvider(BrowserProvider):
    """Local browser provider that launches Playwright Chromium locally."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._cdp_port = None
        self._browser_process = None
        self._xvfb_process = None

    async def launch(self, **kwargs) -> str:
        """Launch a local Chromium browser with CDP endpoint.

        Returns:
            CDP URL for connecting to the browser
        """
        # Start Xvfb for headless display
        await self._start_xvfb()

        # Find free port for CDP
        self._cdp_port = self._find_free_port()

        # Launch Chromium with CDP debugging enabled
        chrome_args = [
            "chromium-browser",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-blink-features=AutomationControlled",
            f"--remote-debugging-port={self._cdp_port}",
            "--window-size=1920,1080",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--no-first-run",
            "--disable-sync",
            "--no-default-browser-check",
            "about:blank",
        ]

        env = os.environ.copy()
        env["DISPLAY"] = ":1"

        self._browser_process = subprocess.Popen(
            chrome_args,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for CDP to be ready
        await self._wait_for_cdp()

        # Return CDP WebSocket URL
        return f"ws://localhost:{self._cdp_port}/devtools/browser"

    async def _start_xvfb(self):
        """Start Xvfb virtual display."""
        # Check if X11 is already running
        if os.path.exists("/tmp/.X11-unix/X1"):
            return

        self._xvfb_process = subprocess.Popen(
            ["Xvfb", ":1", "-screen", "0", "1920x1080x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for X11 to be ready
        for _ in range(50):
            if os.path.exists("/tmp/.X11-unix/X1"):
                break
            await asyncio.sleep(0.1)

    async def _wait_for_cdp(self):
        """Wait for CDP endpoint to be ready."""
        import socket

        for _ in range(100):  # 10 seconds max
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex(("localhost", self._cdp_port))
                sock.close()
                if result == 0:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)

        raise TimeoutError(f"CDP endpoint not ready on port {self._cdp_port}")

    def _find_free_port(self) -> int:
        """Find a free port for CDP."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def close(self) -> None:
        """Close the local browser."""
        if self._browser_process:
            self._browser_process.terminate()
            self._browser_process = None

        if self._xvfb_process:
            self._xvfb_process.terminate()
            self._xvfb_process = None

    def get_live_view_url(self) -> str | None:
        """Return VNC URL for live viewing."""
        return "http://localhost:8080/vnc.html"
'''

    # Write the new implementation
    with open(kernel_provider_path, 'w') as f:
        f.write(new_content)

    print("✅ Created local kernel provider implementation")

def update_docker_config():
    """Update the Docker configuration to use kernel provider."""

    # Update the conversion script
    convert_script = '''#!/usr/bin/env python3
"""Convert SheetBench-50.json to use local Docker with kernel provider"""

import json

def convert_mcp_config():
    # Read the original file
    with open('SheetBench-50.json', 'r') as f:
        data = json.load(f)

    # Docker-based mcp_config with kernel provider (local)
    docker_config = {
        "browser": {
            "command": "docker",
            "args": [
                "run", "--rm", "-i", "-p", "8080:8080",
                "-e", "BROWSER_PROVIDER=kernel",
                "hudevals/hud-remote-browser:0.1.1"
            ]
        }
    }

    # Replace mcp_config in each task
    for task in data:
        if 'mcp_config' in task:
            task['mcp_config'] = docker_config

    # Write back to file
    with open('SheetBench-50-local.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Converted {len(data)} tasks to use local kernel provider")
    print("📁 Saved as: SheetBench-50-local.json")

if __name__ == "__main__":
    convert_mcp_config()
'''

    with open('convert_to_local.py', 'w') as f:
        f.write(convert_script)

    print("✅ Created local conversion script")

if __name__ == "__main__":
    print("🔧 Setting up local browser provider...")
    create_local_kernel_provider()
    update_docker_config()
    print("\n🚀 Setup complete! Now you can run:")
    print("   python convert_to_local.py")
    print("   uv run -m hud eval SheetBench-50-local.json claude --max-steps 100 --per-task --task-log-dir ./my_logs_full_claude_local")