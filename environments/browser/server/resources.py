"""Browser environment MCP resources."""

import json
import os
from datetime import datetime

from hud.server import MCPRouter

# Create router for this module
router = MCPRouter()


@router.resource("telemetry://live")
async def get_telemetry() -> str:
    """MCP resource containing telemetry data including VNC live_url."""
    telemetry_data = {
        "live_url": "http://localhost:8080/vnc.html",
        "display": os.getenv("DISPLAY", ":0"),
        "vnc_port": 8080,
        "websockify_port": 8080,
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "services": {"x11": "running", "vnc": "running", "websockify": "running"},
    }
    return json.dumps(telemetry_data, indent=2)
