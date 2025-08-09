"""Setup functions for managing apps in the browser environment."""

import os
import subprocess
import asyncio
from typing import Dict, Any
from hud.tools.setup import BaseSetup, SetupResult

# Get the global setup tool instance and decorator from __init__.py
from . import setup


@setup("rebuild_app", description="Rebuild and restart an app (frontend and/or backend)")
class RebuildAppSetup(BaseSetup):
    """Setup tool that rebuilds and restarts an app."""

    async def __call__(
        self, context, app_name: str, rebuild_frontend: bool = True, rebuild_backend: bool = True
    ) -> SetupResult:
        """Rebuild and restart an app.

        Args:
            context: The browser environment context
            app_name: Name of the app to rebuild (e.g., 'todo')
            rebuild_frontend: Whether to rebuild the frontend
            rebuild_backend: Whether to rebuild/restart the backend

        Returns:
            SetupResult with rebuild status
        """
        try:
            app_dir = f"/app/apps/{app_name}"
            if not os.path.exists(app_dir):
                return SetupResult(status="error", message=f"❌ App '{app_name}' not found")

            messages = []

            # Rebuild frontend if requested
            if rebuild_frontend:
                frontend_dir = f"{app_dir}/frontend"
                if os.path.exists(frontend_dir):
                    try:
                        # Build the frontend
                        result = subprocess.run(
                            ["npm", "run", "build"],
                            cwd=frontend_dir,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if result.returncode == 0:
                            messages.append(f"✅ Frontend rebuilt successfully")
                        else:
                            messages.append(f"⚠️ Frontend rebuild failed: {result.stderr[:200]}")
                    except Exception as e:
                        messages.append(f"⚠️ Frontend rebuild error: {str(e)}")
                else:
                    messages.append(f"ℹ️ No frontend directory found")

            # Restart backend if requested
            if rebuild_backend:
                backend_dir = f"{app_dir}/backend"
                if os.path.exists(backend_dir):
                    try:
                        # Kill existing backend process if running
                        if hasattr(context.service_manager, "_app_processes"):
                            if app_name in context.service_manager._app_processes:
                                old_proc = context.service_manager._app_processes[app_name]
                                if old_proc and old_proc.poll() is None:
                                    old_proc.terminate()
                                    await asyncio.sleep(0.5)
                                    if old_proc.poll() is None:
                                        old_proc.kill()
                                    messages.append(f"✅ Stopped existing {app_name} backend")

                        # Restart the app through service manager
                        result = await context.service_manager.launch_app(app_name)
                        if "error" not in result.get("status", "").lower():
                            messages.append(f"✅ Backend restarted successfully")
                        else:
                            messages.append(
                                f"⚠️ Backend restart failed: {result.get('message', 'Unknown error')}"
                            )
                    except Exception as e:
                        messages.append(f"⚠️ Backend restart error: {str(e)}")
                else:
                    messages.append(f"ℹ️ No backend directory found")

            # Return combined status
            status = "success" if any("✅" in msg for msg in messages) else "error"
            return SetupResult(status=status, message="\n".join(messages))

        except Exception as e:
            return SetupResult(status="error", message=f"❌ Rebuild failed: {str(e)}")


@setup("restart_all_apps", description="Restart all running apps")
class RestartAllAppsSetup(BaseSetup):
    """Setup tool that restarts all running apps."""

    async def __call__(self, context) -> SetupResult:
        """Restart all running apps.

        Args:
            context: The browser environment context

        Returns:
            SetupResult with restart status
        """
        try:
            # Get list of running apps
            if not hasattr(context.service_manager, "_app_processes"):
                return SetupResult(status="error", message="❌ No apps are currently running")

            running_apps = list(context.service_manager._app_processes.keys())
            if not running_apps:
                return SetupResult(status="error", message="❌ No apps are currently running")

            messages = []
            for app_name in running_apps:
                try:
                    # Kill existing process
                    proc = context.service_manager._app_processes[app_name]
                    if proc and proc.poll() is None:
                        proc.terminate()
                        await asyncio.sleep(0.5)
                        if proc.poll() is None:
                            proc.kill()

                    # Restart the app
                    result = await context.service_manager.launch_app(app_name)
                    if "error" not in result.get("status", "").lower():
                        messages.append(f"✅ Restarted {app_name}")
                    else:
                        messages.append(
                            f"⚠️ Failed to restart {app_name}: {result.get('message', 'Unknown error')}"
                        )
                except Exception as e:
                    messages.append(f"⚠️ Error restarting {app_name}: {str(e)}")

            return SetupResult(
                status="success" if messages else "error",
                message="\n".join(messages) if messages else "❌ No apps restarted",
            )

        except Exception as e:
            return SetupResult(status="error", message=f"❌ Restart all failed: {str(e)}")
