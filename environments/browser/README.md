# Browser Environment

Browser automation environment with GUI access for testing web applications. Includes sample apps (2048, Todo) and supports hot-reload development.

## Architecture

**`environment/`** - Produces structured data
- FastAPI backend with X11/VNC services (Linux-only)
- Launches and manages web apps (Next.js frontends + Python backends)
- Exposes HTTP endpoints for app control and state

**`server/`** - Wraps data in MCP tools
- Browser automation tools (Playwright, computer vision)
- Setup tools (launch apps, seed data)
- Evaluation tools (check game state, todo completion)

**Why separate?** The environment backend requires X11/VNC/Chromium (Docker-only). The MCP server tools can be edited with hot-reload, while the heavy environment stays running.

## Development

This environment **requires Docker** due to X11/VNC dependencies.

```bash
# Build first (creates hud-browser:0.1.0)
hud build

# Start with hot-reload
hud dev
```

When you run `hud dev` in an environment with a Dockerfile, it automatically:
- Detects Docker mode is needed
- Mounts `server/` and `environment/` as volumes
- Enables hot-reload for both layers

Edit files in `server/` or `environment/` and they reload inside the container!

## Publishing Your Environment

Once your environment is ready, you can share it with the community:

### 1. Push to Registry
```bash
# Build and push your environment (requires docker hub login and hud api key)
hud build
hud push
```

### 2. Create a Dataset

Create a dataset on HuggingFace with your tasks:

**Option A: Upload manually**
1. Upload your `tasks.json` to HuggingFace
2. Make sure it's **public** to appear on leaderboards

**Option B: Use the SDK**
```python
from hud.datasets import save_tasks
import json

# Load your tasks
with open("tasks.json") as f:
    tasks = json.load(f)

# Push to HuggingFace
save_tasks(tasks, repo_id="your-org/your-dataset")
```

### 3. Run and Track Performance

```bash
# Run Claude on your benchmark
hud eval "your-org/your-dataset" --agent claude

# View results at:
# hud.ai/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.ai/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.ai/evaluate-agents/leaderboards)

## Architecture Overview

The browser environment uses a two-process architecture:

1. **Context Server** (`context.py`): Long-running process that maintains persistent state
2. **MCP Server** (`server.py`): Hot-reloadable process that handles tool requests

### Key Components

- **BrowserContext**: Stores persistent state (running apps, ports, playwright instance)
- **ServiceManager**: Manages X11, VNC, and app processes
- **BaseHub Tools**: Setup and evaluate tools organized by app (2048, todo)
- **Multiprocessing Proxy**: Enables state sharing between processes

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

