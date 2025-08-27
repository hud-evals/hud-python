# Minetest GUI Environment

Open-source Minecraft-like game wrapped as a HUD MCP environment with real computer control (mouse, keyboard, screenshots) via noVNC.

## Quick Start

```bash
cd environments/minetest
# Build

docker build -t hud-minetest .

# Validate with HUD debug
hud debug hud-minetest
```

## Tools
- `setup` hub
  - `launch(world_name?, fullscreen?)` – launch Minetest
  - `ensure_running()` – ensure process is running
- `evaluate` hub
  - `health()` – basic readiness info
- `computer` – full CLA actions (click, type, press, drag, screenshot, ...)

## Dev with hud dev

```bash
hud dev . --build
```

Open noVNC at http://localhost:8080/vnc.html

## Notes
- Uses system `minetest` package and starts Xvfb + x11vnc + websockify.
- Screenshots provided by XDO/scrot through `HudComputerTool`.