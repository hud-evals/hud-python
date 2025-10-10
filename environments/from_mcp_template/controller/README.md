# Controller

Frontend for the agent: defines tools, minimal state, calls the environment over HTTP.

What to implement
- Shared client in `__init__.py` (one `httpx.AsyncClient`)
- Lifecycle in `hooks.py` (`@mcp.initialize`/`@mcp.shutdown`)
- Tools in `tools.py` (`@mcp.tool`) — keep logic thin; docstrings = descriptions

Run
```bash
hud run controller --transport http --reload
# Helper endpoints: http://localhost:8765/hud and /hud/tools
```

Principle: the controller is UX, not state. Keep long‑lived state in the environment.
