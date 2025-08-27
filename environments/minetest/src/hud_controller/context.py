"""
Persistent context server for Minetest environment.

Maintains service manager and can launch the Minetest process attached to X11.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any

from hud.server.context import run_context_server

logger = logging.getLogger(__name__)


@dataclass
class MinetestConfig:
    world_name: str = "hud_world"
    fullscreen: bool = True
    geometry: str = "1920x1080"


class ServiceManager:
    """Start/monitor X11, VNC, and Minetest process."""

    def __init__(self) -> None:
        self.minitest_proc: Optional[subprocess.Popen] = None

    async def launch_minetest(self, config: MinetestConfig) -> None:
        """Launch Minetest if not running."""
        if self.minitest_proc and self.minitest_proc.poll() is None:
            logger.info("Minetest already running")
            return

        env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")}

        # Build command - use system minetest
        cmd = [
            "minetest",
            "--go",
            "--name", "hud",
            "--worldname", config.world_name,
        ]
        if config.fullscreen:
            cmd += ["--fullscreen"]

        # Start process; do not attach pipes to stdout/stderr
        self.minitest_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        logger.info("Launched Minetest process (PID=%s)", self.minitest_proc.pid)

    def is_minetest_running(self) -> bool:
        return self.minitest_proc is not None and self.minitest_proc.poll() is None

    def shutdown(self) -> None:
        if self.minitest_proc and self.minitest_proc.poll() is None:
            self.minitest_proc.terminate()
            try:
                self.minitest_proc.wait(timeout=3)
            except Exception:
                self.minitest_proc.kill()
        self.minitest_proc = None


class MinetestContext:
    """Context object shared via multiprocessing manager."""

    def __init__(self) -> None:
        self.service_manager = ServiceManager()
        self.is_initialized = False
        self.config = MinetestConfig()

    # Proxy-friendly methods
    def get_service_manager(self) -> ServiceManager:
        return self.service_manager

    def get_is_initialized(self) -> bool:
        return self.is_initialized

    def set_initialized(self, value: bool) -> None:
        self.is_initialized = value

    def get_config(self) -> Dict[str, Any]:
        return {
            "world_name": self.config.world_name,
            "fullscreen": self.config.fullscreen,
            "geometry": self.config.geometry,
        }


if __name__ == "__main__":
    # Ensure DISPLAY is set for downstream processes
    os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":1")

    ctx = MinetestContext()
    logger.info("[MinetestContext] Starting context server")
    asyncio.run(run_context_server(ctx, "/tmp/hud_minetest_ctx.sock"))

