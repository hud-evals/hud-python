# Browser Environment

A browser automation environment for HUD that provides GUI access and web app interaction capabilities. This environment supports hot-reloading during development while maintaining persistent state.

## Architecture Overview

The browser environment uses a two-process architecture:

1. **Context Server** (`context.py`): Long-running process that maintains persistent state
2. **MCP Server** (`server.py`): Hot-reloadable process that handles tool requests

### Key Components

- **BrowserContext**: Stores persistent state (running apps, ports, playwright instance)
- **ServiceManager**: Manages X11, VNC, and app processes
- **BaseHub Tools**: Setup and evaluate tools organized by app (2048, todo)
- **Multiprocessing Proxy**: Enables state sharing between processes

## Context Management and Common Pitfalls

### Understanding the Proxy System

The browser environment uses Python's `multiprocessing.Manager` to share state between the context server and MCP server. This introduces important constraints:

#### ❌ Common Pitfall: Unpicklable Objects

```python
# BAD: This will fail with "cannot pickle 'coroutine' object"
@setup.tool("my_tool")
async def my_tool():
    env = setup.env
    result = await env.call_app_api("app", "/api/endpoint")  # Returns coroutine
    # The coroutine can't be serialized through the proxy!
```

#### ✅ Solution: Direct HTTP Calls

```python
# GOOD: Make HTTP calls directly
@setup.tool("my_tool")
async def my_tool():
    import httpx
    
    # Get the backend port from persistent context
    persistent_ctx = setup.env
    backend_port = persistent_ctx.get_app_backend_port("app")
    
    # Make API call directly
    url = f"http://localhost:{backend_port}/api/endpoint"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        result = response.json()
```

### State Synchronization Issues

#### ❌ Common Pitfall: Direct List/Dict Manipulation

```python
# BAD: Regular Python lists don't sync through proxy
class ServiceManager:
    def __init__(self):
        self._launched_apps = []  # Won't sync!
```

#### ✅ Solution: Store State in Persistent Context

```python
# GOOD: Use the persistent context for shared state
class BrowserContext:
    def __init__(self):
        self._running_apps: List[str] = []
        self._app_ports: Dict[str, Dict[str, int]] = {}
    
    def add_running_app(self, app_name: str) -> None:
        """Add app to running list."""
        if app_name not in self._running_apps:
            self._running_apps.append(app_name)
```

### Accessing Shared Resources

#### ❌ Common Pitfall: Direct Attribute Access

```python
# BAD: Direct attribute access on proxy objects
playwright_tool = env.playwright  # May not work with proxy
```

#### ✅ Solution: Use Getter Methods

```python
# GOOD: Use proxy-friendly getter methods
playwright_tool = persistent_ctx.get_playwright_tool()
```

## Best Practices

### 1. Tool Implementation Pattern

All setup and evaluate tools should follow this pattern:

```python
@setup.tool("tool_name")
async def tool_name(param1: type, param2: type):
    """Tool description."""
    try:
        # Get persistent context
        persistent_ctx = setup.env  # or evaluate.env
        
        # Get app ports
        backend_port = persistent_ctx.get_app_backend_port("app_name")
        
        # Make HTTP request
        url = f"http://localhost:{backend_port}/api/endpoint"
        async with httpx.AsyncClient() as client:
            response = await client.method(url, json=data)
            response.raise_for_status()
            result = response.json()
        
        # Return result
        return TextContent(
            text=f"Success message",
            type="text"
        )
    except Exception as e:
        logger.error(f"tool_name failed: {e}")
        return TextContent(
            text=f"Failed: {str(e)}",
            type="text"
        )
```

### 2. App Launch Pattern

When launching apps, ensure ports are stored in the persistent context:

```python
# In launch_app tool
app_info = await service_manager.launch_app(app_name)

# Store ports in persistent context for later access
try:
    backend_port = service_manager.get_app_port(app_name)
    frontend_port = service_manager.get_app_frontend_port(app_name)
    persistent_ctx.set_app_ports(app_name, frontend_port, backend_port)
except Exception as e:
    logger.error(f"Failed to store ports: {e}")

# Track app in persistent context
persistent_ctx.add_running_app(app_name)
```

### 3. Import Organization

Keep imports at module level:

```python
# At top of file
import logging
import httpx
from mcp.types import TextContent
from . import setup

# Not inside functions
```

## Troubleshooting

### "Cannot pickle 'coroutine' object"

**Cause**: Trying to return an async function result through the proxy.

**Fix**: Don't use async methods on proxied objects. Make direct HTTP calls instead.

### "App not launched" errors

**Cause**: State synchronization issue between ServiceManager and persistent context.

**Fix**: Ensure `launch_app` stores app info in the persistent context, and setup/evaluate tools check the persistent context's app list.

### "Object has no attribute" on proxy objects

**Cause**: Direct attribute access on multiprocessing proxy objects.

**Fix**: Use getter/setter methods instead of direct attribute access.

## Agent Display Configuration

The browser environment supports dynamic image rescaling based on agent display dimensions. When agents specify their preferred display dimensions via environment variables, the computer tools will automatically rescale screenshots to match the agent's expected resolution.

### How It Works

1. **Agent sets environment variables**: Agents can set `AGENT_DISPLAY_WIDTH` and `AGENT_DISPLAY_HEIGHT` environment variables
2. **Automatic rescaling**: Computer tools detect these variables and enable image rescaling
3. **Seamless experience**: Screenshots are automatically resized to match the agent's expected resolution

### Usage

```bash
# Run with agent-specific display dimensions
AGENT_DISPLAY_WIDTH=1024 AGENT_DISPLAY_HEIGHT=768 hud dev . --build

# Or set in environment
export AGENT_DISPLAY_WIDTH=1024
export AGENT_DISPLAY_HEIGHT=768
hud dev . --build
```

## Development Workflow

1. **Start the environment**: `hud dev`
2. **Make changes**: Edit tools in `src/hud_controller/`
3. **Test immediately**: The MCP server hot-reloads automatically
4. **Check logs**: Look for serialization or proxy errors

## Adding New Apps

1. Create app directory in `apps/`
2. Add setup tools in `src/hud_controller/setup/app_name.py`
3. Add evaluate tools in `src/hud_controller/evaluate/app_name.py`
4. Follow the HTTP pattern - no `call_app_api` usage
5. Store app ports in persistent context when launching

## Key Files

- `context.py`: Persistent state management
- `server.py`: MCP server and tool definitions
- `services.py`: Process management for X11, VNC, apps
- `setup/`: Setup tools organized by app
- `evaluate/`: Evaluation tools organized by app

Remember: When in doubt, make direct HTTP calls and store state in the persistent context!

