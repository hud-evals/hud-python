#!/usr/bin/env python3
"""Todo app launcher script."""

import subprocess
import time
import signal
import sys
import argparse
import logging
import os
import socket
from pathlib import Path
from typing import Optional

# Configure logging to stderr to avoid stdio contamination
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] TodoApp: %(message)s", stream=sys.stderr
)

# Global variables to track processes
frontend_process: Optional[subprocess.Popen] = None
backend_process: Optional[subprocess.Popen] = None


def cleanup_processes():
    """Clean up running processes."""
    global frontend_process, backend_process
    logging.info("Shutting down services...")

    for proc in [frontend_process, backend_process]:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    cleanup_processes()
    sys.exit(0)


def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result != 0  # Port is available if connection fails
    except:
        return True


def launch_app(frontend_port: int = 3000, backend_port: int = 5000):
    """Launch the todo app with frontend and backend."""
    global frontend_process, backend_process

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Get current directory
        app_dir = Path(__file__).parent
        frontend_dir = app_dir / "frontend"
        backend_dir = app_dir / "backend"

        logging.info(
            f"Starting todo app - Frontend port: {frontend_port}, Backend port: {backend_port}"
        )

        # Check if ports are available
        if not check_port_available(backend_port):
            logging.warning(f"Backend port {backend_port} is already in use")
        if not check_port_available(frontend_port):
            logging.warning(f"Frontend port {frontend_port} is already in use")

        # Prepare backend command
        backend_env = {
            "PORT": str(backend_port),
            "PYTHONPATH": str(backend_dir),
            **dict(os.environ),
        }

        # Check if we can use uv, otherwise fall back to system python
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            backend_cmd = [
                "uv",
                "run",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(backend_port),
            ]
            logging.info("Using uv for backend")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to system python with uvicorn
            logging.info("uv not available, using system python for backend")
            backend_cmd = [
                "python3",
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(backend_port),
            ]

        # Prepare frontend command
        frontend_env = {
            "NEXT_PUBLIC_API_URL": f"http://localhost:{backend_port}",
            "PORT": str(frontend_port),
            **dict(os.environ),
        }

        # Check if dependencies are installed
        if frontend_dir.exists():
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                logging.info("Installing frontend dependencies...")
                npm_install = subprocess.run(
                    ["npm", "install"], cwd=frontend_dir, capture_output=True
                )
                if npm_install.returncode != 0:
                    logging.error(
                        f"Failed to install npm dependencies: {npm_install.stderr.decode()}"
                    )
                    cleanup_processes()
                    raise RuntimeError("npm install failed")

            # Check if we have a production build
            if (frontend_dir / ".next").exists():
                logging.info("Running in production mode (pre-built)...")
                frontend_cmd = [
                    "npm",
                    "run",
                    "start",
                    "--",
                    "--port",
                    str(frontend_port),
                    "--hostname",
                    "0.0.0.0",
                ]
            else:
                logging.info("Running in development mode...")
                frontend_cmd = [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--port",
                    str(frontend_port),
                    "--hostname",
                    "0.0.0.0",
                ]

        # 🚀 START BOTH PROCESSES IN PARALLEL
        logging.info("Starting backend and frontend in parallel...")

        # Start backend - UPDATE GLOBAL VARIABLE
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=backend_dir,
            env=backend_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # Don't capture stdout - reserved for MCP
            stderr=subprocess.DEVNULL,  # Don't capture stderr - reserved for MCP
        )

        # Start frontend immediately (in parallel) - UPDATE GLOBAL VARIABLE
        if frontend_dir.exists():
            frontend_process = subprocess.Popen(
                frontend_cmd,
                cwd=frontend_dir,
                env=frontend_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,  # Don't capture stdout - reserved for MCP
                stderr=subprocess.DEVNULL,  # Don't capture stderr - reserved for MCP
            )

        # 🚀 WAIT FOR BOTH IN PARALLEL WITH FAST POLLING
        backend_ready = False
        frontend_ready = False

        # Use faster polling (every 200ms instead of 1s)
        max_attempts_backend = 150  # 30 seconds at 200ms intervals
        max_attempts_frontend = 600  # 120 seconds at 200ms intervals

        for attempt in range(max(max_attempts_backend, max_attempts_frontend)):
            # Check if processes are still alive
            if backend_process and backend_process.poll() is not None:
                logging.error(f"Backend process died with exit code {backend_process.returncode}")
                cleanup_processes()
                raise RuntimeError("Backend failed to start")

            if frontend_process and frontend_process.poll() is not None:
                logging.error(f"Frontend process died with exit code {frontend_process.returncode}")
                cleanup_processes()
                raise RuntimeError("Frontend failed to start")

            # Check backend readiness
            if not backend_ready and attempt < max_attempts_backend:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                try:
                    result = sock.connect_ex(("localhost", backend_port))
                    sock.close()
                    if result == 0:
                        backend_ready = True
                        logging.info(f"Backend is ready (attempt {attempt + 1})")
                except:
                    pass

            # Check frontend readiness
            if not frontend_ready and attempt < max_attempts_frontend:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                try:
                    result = sock.connect_ex(("localhost", frontend_port))
                    sock.close()
                    if result == 0:
                        frontend_ready = True
                        logging.info(f"Frontend is ready (attempt {attempt + 1})")
                except:
                    pass

            # Exit early if both are ready
            if backend_ready and frontend_ready:
                break

            time.sleep(0.2)  # 200ms intervals instead of 1s

        # Check final status
        if not backend_ready:
            logging.error("Backend did not start within 30 seconds")
            cleanup_processes()
            raise RuntimeError("Backend startup timeout")

        if not frontend_ready:
            logging.error("Frontend did not start within 2 minutes")
            cleanup_processes()
            raise RuntimeError("Frontend startup timeout")

        # Log startup information
        logging.info("Todo app started successfully!")
        logging.info(f"Frontend: http://localhost:{frontend_port}")
        logging.info(f"Backend API: http://localhost:{backend_port}/docs")
        logging.info("Press Ctrl+C to stop")

        # Wait for processes to finish
        while True:
            time.sleep(1)
            if backend_process and backend_process.poll() is not None:
                logging.error("Backend process died unexpectedly")
                break
            if frontend_process and frontend_process.poll() is not None:
                logging.error("Frontend process died unexpectedly")
                break

    except Exception as e:
        logging.error(f"Error launching app: {e}")
        cleanup_processes()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Todo App")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port")
    parser.add_argument("--backend-port", type=int, default=5000, help="Backend port")

    args = parser.parse_args()

    try:
        launch_app(args.frontend_port, args.backend_port)
    except KeyboardInterrupt:
        logging.info("App interrupted by user")
    except Exception as e:
        logging.error(f"Failed to launch app: {e}")
        sys.exit(1)
