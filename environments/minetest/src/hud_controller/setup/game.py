"""Setup tools for launching and configuring Minetest."""

import logging
from typing import Optional

from . import setup
from hud.tools.types import SetupResult

logger = logging.getLogger(__name__)


@setup.tool()
async def launch(world_name: Optional[str] = None, fullscreen: bool = True) -> SetupResult:
    """Launch Minetest with desired configuration.

    Args:
        world_name: Optional world name to create/open
        fullscreen: Whether to launch in fullscreen
    """
    ctx = setup.env
    svc = ctx.get_service_manager()

    # Update context config if provided
    cfg = ctx.get_config()
    if world_name:
        cfg["world_name"] = world_name
    cfg["fullscreen"] = bool(fullscreen)

    # Construct config object for launch
    from ..context import MinetestConfig

    config = MinetestConfig(
        world_name=cfg.get("world_name", "hud_world"),
        fullscreen=cfg.get("fullscreen", True),
        geometry=cfg.get("geometry", "1920x1080"),
    )

    await svc.launch_minetest(config)

    return SetupResult(content="Minetest launched", info={"running": svc.is_minetest_running()})


@setup.tool()
async def ensure_running() -> SetupResult:
    """Ensure Minetest process is running."""
    ctx = setup.env
    svc = ctx.get_service_manager()
    if svc.is_minetest_running():
        return SetupResult(content="Minetest already running", info={"running": True})

    # If not running, launch with defaults
    from ..context import MinetestConfig

    await svc.launch_minetest(MinetestConfig())
    return SetupResult(content="Minetest started", info={"running": svc.is_minetest_running()})

