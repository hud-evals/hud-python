"""
FastAPI server for browser environment.
Exposes API endpoints to interact with the environment and its subcomponents.
"""

import asyncio
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Set
import socket
from contextlib import asynccontextmanager
import shutil
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class AppInfo(BaseModel):
    """Information about a launched app."""

    name: str
    frontend_port: int
    backend_port: int
    url: str
    status: str


class ServiceStatus(BaseModel):
    """Status of environment services."""

    x11: bool
    vnc: bool
    websockify: bool
    apps: List[AppInfo]


class LaunchAppRequest(BaseModel):
    """Request to launch an app."""

    app_name: str


class LaunchAppResponse(BaseModel):
    """Response after launching an app."""

    name: str
    url: str
    frontend_port: int
    backend_port: int


class ServiceManager:
    """Manages environment services (X11, VNC, apps)."""

    def __init__(self):
        self.x11_proc: Optional[subprocess.Popen] = None
        self.vnc_proc: Optional[subprocess.Popen] = None
        self.websockify_proc: Optional[subprocess.Popen] = None
        self.chrome_proc: Optional[subprocess.Popen] = None
        self.cdp_port: Optional[int] = None
        self._launched_apps: Dict[str, AppInfo] = {}
        self._playwright = None
        self._browser = None
        self._app_processes: Dict[str, subprocess.Popen] = {}
        self._allocated_ports: Set[int] = set()

    async def start_core_services(self):
        """Start X11, VNC, and websockify services."""
        # Check if X11 is already running
        if Path("/tmp/.X11-unix/X1").exists():
            logger.info("X11 display :1 already running")
        else:
            # Start Xvfb if not already running
            self.x11_proc = subprocess.Popen(
                ["Xvfb", ":1", "-screen", "0", "1920x1080x24"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info("Started Xvfb on display :1")

        # Wait for X11
        await self._wait_for_x11()

        # Start VNC and websockify
        await self._start_vnc_services()

    async def _wait_for_x11(self):
        """Wait for X11 display to be ready."""
        for i in range(100):  # 10 seconds max
            if Path("/tmp/.X11-unix/X1").exists():
                logger.info("X11 display :1 is ready")
                os.environ["DISPLAY"] = ":1"
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("X11 failed to start")

    async def _start_vnc_services(self):
        """Start VNC and websockify services."""
        # Start x11vnc
        self.vnc_proc = subprocess.Popen(
            ["x11vnc", "-display", ":1", "-forever", "-shared", "-nopw"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ, "DISPLAY": ":1"},
        )
        logger.info("Started x11vnc")

        # Start websockify
        self.websockify_proc = subprocess.Popen(
            ["websockify", "--web", "/usr/share/novnc", "8080", "localhost:5900"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("Started websockify on port 8080")

        # Wait for both services
        await asyncio.gather(
            self._wait_for_port(5900, "VNC"), self._wait_for_port(8080, "websockify")
        )
        logger.info("noVNC available at: http://localhost:8080/vnc.html")

        # Start Playwright's Chromium browser
        logger.info("Starting Playwright's Chromium browser")
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            # Get a free port for CDP
            self.cdp_port = self._get_next_port()

            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=[
                    f"--remote-debugging-port={self.cdp_port}",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--display=:1",
                    "--start-maximized",
                ],
                env={**os.environ, "DISPLAY": ":1"},
            )

            logger.info(f"Started Playwright Chromium with CDP on port {self.cdp_port}")

            # Wait for CDP to be ready
            await self._wait_for_port(self.cdp_port, "CDP", timeout=30)

            # Open a default page so the browser window is visible
            default_context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080}, no_viewport=False
            )
            default_page = await default_context.new_page()
            await default_page.goto("about:blank")
            logger.info("Opened default browser page")

        except ImportError:
            logger.error("Playwright not installed")
            raise RuntimeError("Playwright is required. The Docker image should have installed it.")
        except Exception as e:
            logger.error(f"Failed to start Playwright browser: {e}")
            raise

    async def launch_app(self, app_name: str) -> LaunchAppResponse:
        """Launch a specific app dynamically."""
        # Check if app is already running
        if app_name in self._launched_apps:
            app_info = self._launched_apps[app_name]
            if app_info.status == "running":
                return LaunchAppResponse(
                    name=app_info.name,
                    url=app_info.url,
                    frontend_port=app_info.frontend_port,
                    backend_port=app_info.backend_port,
                )

        app_path = Path(f"/app/environment/{app_name}")
        if not app_path.exists():
            raise ValueError(f"App '{app_name}' not found at {app_path}")

        # Check if app has a launch script
        launch_script = app_path / "launch.py"
        if not launch_script.exists():
            raise ValueError(f"App '{app_name}' missing launch.py")

        # Get unique ports for frontend and backend
        frontend_port = self._get_next_port()
        backend_port = self._get_next_port()

        # Launch the app
        proc = subprocess.Popen(
            [
                "python3",
                str(launch_script),
                "--frontend-port",
                str(frontend_port),
                "--backend-port",
                str(backend_port),
            ],
            cwd=app_path,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "DISPLAY": ":1"},
        )

        self._app_processes[app_name] = proc

        try:
            # Wait for both ports
            await asyncio.gather(
                self._wait_for_port(frontend_port, f"app '{app_name}' frontend", timeout=60),
                self._wait_for_port(backend_port, f"app '{app_name}' backend", timeout=60),
            )

            logger.info(
                f"Launched app '{app_name}' - Frontend: {frontend_port}, Backend: {backend_port}"
            )

            # Store app information
            app_info = AppInfo(
                name=app_name,
                frontend_port=frontend_port,
                backend_port=backend_port,
                url=f"http://localhost:{frontend_port}",
                status="running",
            )
            self._launched_apps[app_name] = app_info

            return LaunchAppResponse(
                name=app_name,
                url=app_info.url,
                frontend_port=frontend_port,
                backend_port=backend_port,
            )

        except TimeoutError:
            # Check if process is still running
            if proc.poll() is not None:
                logger.error(f"App '{app_name}' process exited with code {proc.returncode}")
            else:
                logger.error(f"App '{app_name}' failed to become ready within timeout")
            raise

    def get_service_status(self) -> ServiceStatus:
        """Get status of all services."""
        # Update app statuses
        for app_name, proc in self._app_processes.items():
            if app_name in self._launched_apps:
                if proc.poll() is None:
                    self._launched_apps[app_name].status = "running"
                else:
                    self._launched_apps[app_name].status = "stopped"

        return ServiceStatus(
            x11=self.x11_proc is not None and self.x11_proc.poll() is None
            if self.x11_proc
            else Path("/tmp/.X11-unix/X1").exists(),
            vnc=self.vnc_proc is not None and self.vnc_proc.poll() is None
            if self.vnc_proc
            else self._is_port_open(5900),
            websockify=self.websockify_proc is not None and self.websockify_proc.poll() is None
            if self.websockify_proc
            else self._is_port_open(8080),
            apps=list(self._launched_apps.values()),
        )

    def get_app_info(self, app_name: str) -> AppInfo:
        """Get information about a specific app."""
        if app_name not in self._launched_apps:
            raise ValueError(f"App '{app_name}' not found")
        return self._launched_apps[app_name]

    async def shutdown(self):
        """Shutdown all services gracefully."""
        # Stop app processes
        for name, proc in self._app_processes.items():
            if proc.poll() is None:
                proc.terminate()
                await asyncio.sleep(1)
                if proc.poll() is None:
                    proc.kill()
                logger.info(f"Terminated app '{name}'")

        # Clear app tracking
        self._app_processes.clear()
        self._launched_apps.clear()
        self._allocated_ports.clear()

        # Close Playwright browser
        if self._browser:
            try:
                await self._browser.close()
                logger.info("Closed Playwright browser")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

        if self._playwright:
            try:
                await self._playwright.stop()
                logger.info("Stopped Playwright")
            except Exception as e:
                logger.error(f"Error stopping playwright: {e}")

        # Stop services in reverse order
        for proc, name in [
            (self.websockify_proc, "websockify"),
            (self.vnc_proc, "x11vnc"),
            (self.x11_proc, "Xvfb"),
        ]:
            if proc and proc.poll() is None:
                proc.terminate()
                await asyncio.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                logger.info(f"Stopped {name}")

    def _is_port_open(self, port: int) -> bool:
        """Check if a port is open."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        try:
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except:
            return False

    def _get_next_port(self) -> int:
        """Get next available port for apps."""
        base_port = 3000
        for offset in range(200):  # Support up to 200 ports
            port = base_port + offset
            if not self._is_port_open(port) and port not in self._allocated_ports:
                self._allocated_ports.add(port)
                return port
        raise RuntimeError("No available ports")

    async def _wait_for_port(self, port: int, service_name: str = "service", timeout: int = 30):
        """Wait for a port to become available."""
        for _ in range(timeout * 5):  # Check every 200ms
            if self._is_port_open(port):
                logger.info(f"{service_name} is ready on port {port}")
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(f"Port {port} did not become available for {service_name}")

    async def get_cdp_websocket_url(self) -> str | None:
        """Discover the actual CDP WebSocket URL from Chrome's /json/version endpoint."""
        if not self.cdp_port:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.cdp_port}/json/version", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    # Chrome returns webSocketDebuggerUrl in /json/version response
                    websocket_url = data.get("webSocketDebuggerUrl")
                    if websocket_url:
                        return websocket_url

                # Fallback: try /json/list to find a browser target
                response = await client.get(
                    f"http://localhost:{self.cdp_port}/json/list", timeout=5.0
                )
                if response.status_code == 200:
                    targets = response.json()
                    # Look for a browser target (type 'page' or title containing 'about:blank')
                    for target in targets:
                        if target.get("type") == "page" or "about:blank" in target.get("url", ""):
                            websocket_url = target.get("webSocketDebuggerUrl")
                            if websocket_url:
                                return websocket_url

        except Exception as e:
            logger.warning(f"Failed to discover CDP WebSocket URL: {e}")

        # Final fallback to generic path (may not work)
        return f"ws://localhost:{self.cdp_port}/devtools/browser"


# Global service manager instance
service_manager = ServiceManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting browser environment server...")
    await service_manager.start_core_services()
    logger.info("Browser environment server ready")

    yield

    # Shutdown
    logger.info("Shutting down browser environment server...")
    await service_manager.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Browser Environment API",
    description="API for managing browser environment services and applications",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status", response_model=ServiceStatus)
async def get_status():
    """Get status of all environment services."""
    return service_manager.get_service_status()


@app.post("/apps/launch", response_model=LaunchAppResponse)
async def launch_app(request: LaunchAppRequest):
    """Launch a specific application."""
    try:
        return await service_manager.launch_app(request.app_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/apps/{app_name}", response_model=AppInfo)
async def get_app_info(app_name: str):
    """Get information about a specific app."""
    try:
        return service_manager.get_app_info(app_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/vnc/url")
async def get_vnc_url():
    """Get the VNC viewer URL."""
    return {"url": "http://localhost:8080/vnc.html"}


@app.get("/display")
async def get_display():
    """Get the X11 display information."""
    return {
        "display": os.environ.get("DISPLAY", ":1"),
        "x11_running": Path("/tmp/.X11-unix/X1").exists(),
    }


@app.get("/cdp")
async def get_cdp():
    """Return the CDP websocket URL for connecting Playwright/Chromium clients."""
    if service_manager.cdp_port is None:
        raise HTTPException(status_code=503, detail="CDP not available")

    # Discover the actual CDP WebSocket URL from Chrome
    websocket_url = await service_manager.get_cdp_websocket_url()
    if not websocket_url:
        raise HTTPException(status_code=503, detail="CDP WebSocket URL not available")

    return {"ws": websocket_url}


@app.post("/shutdown")
async def shutdown_env():
    """Gracefully stop services and request server shutdown."""
    try:
        await service_manager.shutdown()
    except Exception as e:
        logger.warning(f"Error during environment shutdown: {e}")
    # Signal uvicorn to exit via lifespan shutdown
    # FastAPI/uvicorn doesn't expose server here; we rely on process signal from caller.
    return {"status": "shutting_down"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
