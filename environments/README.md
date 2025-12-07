# How to Build HUD-Compatible MCP Environments

This document is a step-by-step guide for turning *any* piece of software that can run in a Docker container into a **Model Context Protocol (MCP)** environment that the HUD SDK can evaluate or control.  We’ll move through six short phases, each with a clear checkpoint.

> **Big picture**
> • An *agent* (LLM) wants to solve tasks inside a *software environment*.
> • Your job: give that environment a clean, programmable surface – a set of
>   *tools* the agent can invoke.
> • MCP is simply the wire-format we use to move those tool calls back and forth
>   (like gRPC or HTTP but JSON-RPC over stdio/Docker).
> • FastMCP is the underlying SDK; HUD provides **MCPServer** – a thin wrapper that
>   adds SIGTERM handling, `@initialize` / `@shutdown` decorators, and easier
>   tool registration while remaining 100 % compatible with FastMCP.
> 
> The picture:
> ```text
>  LLM Agent ──JSON-RPC──► FastMCP server (your code) ──► real app / game / browser
> ```
> Your job is to wrap *any* app in an MCP server so agents can control it reproducibly & safely.

---

## Phase Overview

| Phase | Goal |
|-------|------|
| 1 | A Docker image that *starts* and prints to **stderr** |
| 2 | A minimal MCP server that responds to `initialize` over **stdio** |
| 3 | Working `setup`, `evaluate`, and **interaction** tools |
| 4 | Image launches remotely on the HUD platform & exposes live telemetry |
| 5 | Fast local iteration with `hud dev` hot-reload |

Take the phases one at a time; do **not** jump ahead.  Each stage's checkpoint is the foundation for the next.

## Reference Implementations

This repository includes two complete MCP environment implementations that demonstrate different levels of complexity:

### 1. `text_2048` - Simple Game Environment
A minimalist ASCII-based 2048 game that showcases:
- Basic hub pattern with setup/evaluate tools
- Custom interaction tools (move command)
- Clean separation of game logic and MCP server
- Minimal dependencies (Python only)
- Perfect for learning the core concepts

### 2. `remote_browser` - Advanced Browser Automation
A sophisticated browser automation environment featuring:
- Multiple cloud browser provider integrations (AnchorBrowser, Steel, BrowserBase, HyperBrowser, Kernel)
- Both Playwright and computer tools for interaction
- Extensive setup/evaluate capabilities (navigation, cookies, sheets, element checks)
- Live telemetry with browser viewing URLs
- Production-ready error handling and cleanup

💡 **Follow along with text_2048** as you work through each phase - it demonstrates all the core patterns with minimal complexity.

### Installing the HUD CLI

The HUD SDK includes a powerful CLI for debugging and analyzing MCP environments:

```bash
# Install HUD CLI globally with uv (recommended)
uv tool install hud-python@latest --python 3.12

# Or use without installing
uvx --from hud-python hud --help

# Verify installation
hud --help
```

Common commands:
```bash
# Debug your Docker image (runs 5-phase test)
hud debug my-mcp-server:latest

# Analyze available tools and resources
hud analyze my-mcp-server:latest --format json

# Debug any command-based MCP server
hud debug --command "python my_server.py"
```
While you move through the phases it's handy to run the **interactive checker** to make sure nothing broke:

```bash
# First build your Docker image
docker build -t my-environment environments/my-environment

# Then debug it
hud debug my-environment
```

**What's the difference?**
- **`hud debug`** - Tests your environment in 5 phases, checking startup, MCP protocol, tools, and readiness. Use this first!
- **`hud analyze`** - Explores the environment to discover all tools, resources, and capabilities. Only works after debug passes phase 3.

The script walks the *same* checklist and prints coloured, human-friendly hints whenever something fails.

| What it validates | Phase |
|-------------------|-------|
| Container starts & logs to **stderr** | 1 |
| MCP server responds to an `initialize` request | 2 |
| Discovers `setup`, `evaluate`, and interaction tools | 3 |
| Calls `setup` / `evaluate`, checks telemetry & startup time | 4 |
| Spawns three concurrent clients to stress-test resources | 5 |

💡 **Run it after finishing each phase.** If the checker exits with a red ❌, scroll up for the gold-coloured *hint* block – it usually points directly to the root cause.

---

## Phase 1 – Write a Dockerfile

**Goal →** Create a container that can run your MCP server with proper Python packaging.

Key principles:
- **stdout** is reserved for MCP protocol (JSON-RPC)
- **stderr** is for all logs and debug output
- Use proper Python packaging with `pyproject.toml`
- Run as a module for clean imports

### Dockerfile Template

```dockerfile
FROM python:3.11-slim

# Prevent Python from buffering output (important for logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy package files
COPY pyproject.toml ./
COPY src/ ./src/

# Install in editable mode for development flexibility
RUN pip install --no-cache-dir -e .

# Run as a module to ensure proper package imports
CMD ["python", "-m", "my_module.server"]
```

### Build & Test

```bash
docker build -t my-environment .

# Test Phase 1: Container should start without errors
docker run --rm -i my-environment
```

### Recommended Environment Structure

For Python-based MCP environments, use this standard structure:

```
my-environment/
├── Dockerfile
├── README.md
├── server/                 # MCP server package
│   ├── pyproject.toml      # MCP dependencies (hud-python, etc.)
│   ├── __init__.py         # Empty package marker
│   ├── main.py             # mcp = MCPServer() + lifecycle hooks
│   ├── tools.py            # router = MCPRouter() + @router.tool decorators
│   ├── setup/              # Setup router (modular approach)
│   │   ├── __init__.py
│   │   ├── basic.py        # Basic setup functions
│   │   └── advanced.py     # Advanced setup functions
│   └── evaluate/           # Evaluate router (modular approach)
│       ├── __init__.py
│       ├── checks.py       # Basic evaluation checks
│       └── metrics.py      # Advanced metrics evaluators
└── environment/            # Backend service package
    ├── pyproject.toml      # Backend dependencies (fastapi, uvicorn)
    ├── __init__.py
    └── server.py           # FastAPI app with /health, /act, /reset, /state
```

This structure enables:
- Clean separation of concerns (environment logic, tools, setup, evaluation)
- Easy volume mounting for development (Phase 5)
- Standard Python packaging with `pip install -e .`
- Modular organization - each setup/evaluator in its own file for clarity

• **One Dockerfile only** – no docker-compose.  
• If you're building a GUI environment, start from `hudpython/novnc-base:latest` instead and leave VNC configuration for later phases.

Checkpoint reached?  Congratulations – move on.

👉 Quick sanity check: `hud debug my-environment` (verifies Phase 1 automatically)

Need inspiration? Check out our reference implementations:
• [`text_2048/Dockerfile`](./text_2048/Dockerfile) - Minimal Python setup, perfect for simple environments
• [`remote_browser/Dockerfile`](./remote_browser/Dockerfile) - Uses pre-built base image with browser dependencies
• [`browser/Dockerfile`](./browser/Dockerfile) - Multi-stage build with full GUI support

---

## Phase 2 – Create the MCP Server

**Goal →** a Python process that:
1. Speaks MCP over **stdio**.
2. Responds correctly to the `initialize` request.
3. Logs everything to **stderr**.

The MCP lifecycle is *initialize → operate → shutdown* (see spec link above).

### Skeleton server (MCPServer)

```python
import sys
import logging
from hud.server import MCPServer

# 1️⃣  Always log to stderr – stdout is reserved for JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s'
)

# Create the server early so decorators can reference it
mcp = MCPServer(name="My Environment")

# Run heavy one-time setup during MCP initialize
@mcp.initialize
async def initialize_environment(session=None, progress_token=None):
    """Heavy one-time setup – start databases, launch background apps, etc."""
    logging.info("starting core services…")
    await start_services()            # your coroutine
    logging.info("services ready")

if __name__ == "__main__":
    mcp.run()
```

*(Replace `start_services()` with whatever takes noticeable startup time – browsers, DBs, X servers, …)*

### Adapt Dockerfile

At the end of your Dockerfile, you must launch the MCP server as the container's main process, ensuring it communicates over stdio (stdin/stdout). This is typically done by setting the `CMD` or `ENTRYPOINT` to run your server module directly, for example:


```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

# Optional: install requirements
# RUN pip install -r requirements.txt

CMD ["python", "-m", "your_module_name"]  # Replace 'your_module_name' with your actual entrypoint module
```

### Three validation steps (run them **in order**)

| # | What you do | Why it matters |
|---|-------------|----------------|
| 1 | **Direct stdio test** – pipe the JSON below into your script | Proves the Python code handles `initialize` without any client or Docker noise |
| 2 | **MCP Inspector** – `npx @modelcontextprotocol/inspector python -m my_package.server` | Lets you click around: view capabilities, tools, resources |
| 3 | **Inside Docker** – rebuild the image and run it | This is *exactly* how HUD will execute the server |
| 4 | **Run `hud debug`** – `hud debug my-environment` | Combines the above checks & points out common mistakes |

#### JSON for step 1

```jsonc
{ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
  "protocolVersion": "2024-11-05",
  "capabilities": {"roots": {"listChanged": true}},
  "clientInfo": {"name": "DevClient", "title": "Dev", "version": "0.0.0"}
}}
```

Pipe it:

```bash
echo '<the-json-above>' | python -m my_package.server
```

If all three validations succeed, you have a real MCP server – time to make it useful.

---

## Phase 3 – Add Setup / Evaluate / Interaction Tools

**Goal →** tools are discoverable in the Inspector *and* callable from the HUD SDK.

👉 After wiring in the tools, confirm with `hud debug my-environment --max-phase 3` – it now checks for their presence and basic execution.

🔍 Once debug passes phase 3, you can analyze the environment:
```bash
hud analyze my-environment  # Interactive view of tools and resources
hud analyze my-environment --format json  # JSON output for scripts
hud analyze my-environment --format markdown  # Generate documentation
```

1. Write **`setup`** and **`evaluate`** tools first – they are *lifecycle* tools and never shown to the LLM.
2. Register at least one **interaction** tool (`computer`, `playwright`, or your own).

### Approach 1: Simple Direct Implementation

For simple environments with just a few setup/evaluate functions, you can use direct tool decorators with **MCPServer**:

```python
from hud.server import MCPServer
from hud.tools import HudComputerTool

mcp = MCPServer(name="my-environment")

@mcp.tool()
async def setup(config: dict) -> dict:
    ...               # prepare environment

@mcp.tool()
async def evaluate(config: dict) -> dict:
    ...               # return {"reward": <0-1>, "done": bool}

@mcp.initialize
async def initialize_environment(session=None, progress_token=None):
    custom_tool = HudComputerTool()
    mcp.add_tool(custom_tool.mcp)
    
    # Any other initialization
```

### Approach 2: Hub Pattern (Recommended for Complex Environments)

The BaseHub pattern provides a clean way to organize multiple setup/evaluate functions with automatic discovery and registration. **A BaseHub is fundamentally another MCP server (it's a subclass of FastMCP)** that you mount to your main server, providing namespace separation and modular organization. All hub functions are exposed through one tool named after the hub, and a resource that can list all of its tools.

When mounted, the hub's tools are accessible through a single tool that dispatches to the appropriate function:
```json
{
  "name": "setup",
  "arguments": {
    "name": "reset",  // Which function in the hub to call
    "arguments": {"param": "value"}  // Additional parameters
  }
}
```

```python
# In setup/__init__.py
from hud.tools.base import BaseHub

# Create the setup hub (a sub-server)
setup = BaseHub("setup")

# Import all setup modules to register their tools
from . import basic, advanced  # This registers all @setup.tool() decorated functions

# In setup/basic.py
from . import setup
from mcp.types import TextContent

@setup.tool()
async def reset(**kwargs):
    """Reset the environment to its initial state.
    
    Args:
        **kwargs: Additional parameters
    
    Returns:
        TextContent
    """
    # Access environment from the hub
    env = setup.env
    await env.reset_state()
    return TextContent(
        text="Environment reset to initial state",
        type="text"
    )

@setup.tool()
async def seed_data(num_items: int = 5):
    """Seed the environment with test data.
    
    Args:
        num_items: Number of items to create
    
    Returns:
        TextContent
    """
    # Access environment from the hub
    env = setup.env
    items = await env.create_items(num_items)
    return TextContent(
        text=f"Created {len(items)} items",
        type="text"
    )

# In evaluate/__init__.py
from hud.tools.base import BaseHub

# Create the evaluate hub (another sub-server)
evaluate = BaseHub("evaluate")

# Import all evaluator modules
from . import checks, metrics

# In evaluate/checks.py
from . import evaluate
from hud.tools.types import EvaluationResult

@evaluate.tool()
async def task_complete(expected_count: int):
    """Check if the expected number of tasks are completed.
    
    Args:
        expected_count: Expected number of completed tasks
    
    Returns:
        EvaluationResult
    """
    # Access environment from the hub
    env = evaluate.env
    completed = await env.count_completed()
    return EvaluationResult(
        reward=min(completed / expected_count, 1.0),
        done=completed >= expected_count,
        content=f"Completed {completed}/{expected_count} tasks",
        info={"completed": completed, "expected": expected_count}
    )

# In server.py
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Create MCP server
mcp = MCPServer(name="my-environment")

@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the environment with progress notifications."""
    # Extract progress token from context
    progress_token = getattr(ctx.meta, "progressToken", None) if ctx.meta else None
    # Send progress updates if available
    async def send_progress(progress: int, message: str):
        if progress_token:
            await ctx.session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=100,
                message=message,
            )
    
    await send_progress(10, "Starting environment initialization...")
    
    # Initialize your environment state/context
    env = await create_environment_context()
    await send_progress(50, "Environment created...")
    
    # Set environment on hubs
    setup_hub.env = env
    evaluate_hub.env = env
    
    # Mount hubs to MCP server
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)
    await send_progress(80, "Tools registered...")
    
    # Register any custom interaction tools
    if hasattr(env, 'custom_tool'):
        mcp.add_tool(env.custom_tool.mcp)
    
    await send_progress(100, "Environment ready!")
```

The BaseHub pattern provides:
- **Namespace isolation**: Tools are grouped under the hub's name (e.g., "setup", "evaluate")
- **Modular organization**: Each hub can be developed and tested independently
- **Type safety**: Full type hints preserved for parameters and returns

When you call a hub's tool, you specify which function to execute:
```python
# Calling the "reset" function in the setup hub
await client.call_tool("setup", {"name": "reset"})

# Calling the "task_complete" function in the evaluate hub  
await client.call_tool("evaluate", {"name": "task_complete", "expected_count": 5})
```

### Test workflow

1. **Inspector first** – restart the server, refresh the *Tools* tab, confirm the new tools appear.  
2. **Run `hud debug my-environment`** – this validates initialization, tool discovery and basic calls automatically.  
3. **Rebuild the image** – `docker build -t my-environment .`.  
4. **HUD SDK script test** – run a short script like the one below.  GUI environments built from `hudpython/novnc-base` still expose a VNC viewer on <http://localhost:8080/vnc.html> – keep it open while testing.

```python
import asyncio
import hud
from hud.datasets import Task
from hud.agents import ClaudeAgent
from hud.clients import MCPClient

async def main():
    # `trace` captures *everything* that happens and sends it to hud.ai
    async with hud.async_trace("local_test"):
        task = Task(
            prompt="Complete the task",
            mcp_config={
                "local": {
                    "command": "docker", 
                    "args": ["run", "--rm", "-i", "my-environment:latest"]
                }
            },
            setup_tool={"name": "setup", "arguments": {"name": "todo_seed", "num_items": 5}},
            evaluate_tool={"name": "evaluate", "arguments": {"name": "todo_completed", "expected_count": 2}}
        )
        client = MCPClient(mcp_config=task.mcp_config)

        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["computer"]  # or ["move"] for text_2048
        )

        result = await agent.run(task)
        print(result)

    await client.close()

asyncio.run(main())
```

The `trace` context manager sends a full timeline of agent actions, tool calls, and rewards to hud.ai – perfect for debugging.

See `examples/01_hello_2048.py` and `examples/task_with_setup_eval.py` for larger end-to-end demos.

---

## Phase 4 – Remote Deployment & HUD Runner

**Goal →** the exact same image runs in parallel on hundreds of instances, and exposes more telemetry so the hud.ai can visualise the whole lifecycle.

### 1. Publish your image

Log in to Docker Hub (or any registry HUD can pull from) and push a tagged build:

```bash
docker tag my-environment yourdockerhubuser/my-environment:latest
docker push yourdockerhubuser/my-environment:latest
```

*(If you’re using a private registry, make sure the HUD worker has pull credentials.)*

### 2. Launch it remotely (gmail_remote pattern)

Here's how to configure a remote MCP server that runs **the same Docker image**:

```python
from hud import settings
from hud.clients import MCPClient

# Your image is in a registry, now tell HUD to pull & run it on demand
config = {
    "hud": {
        "url": settings.hud_mcp_url,
        "headers": {
            "Authorization": f"Bearer {settings.api_key}",
            "Mcp-Image": "yourdockerhubuser/my-environment:latest",  # which image to launch
        },
    }
}

client = MCPClient(mcp_config=config)
```

_Steps 3 and 4 below are **optional but highly recommended** once the image boots successfully._

Spin up **many** agents in parallel by just launching multiple tasks – HUD will queue and start as many containers as resources allow.

### 3. Progress updates during `initialize` (Optional)

At remote scale it can take 10-30 s for heavy services to boot.  Use the new
`@mcp.initialize` decorator plus the `session` / `progress_token` parameters to
stream status messages:

```python
@mcp.initialize
async def initialize_environment(session=None, progress_token=None):
    async def send(p, msg):
        if session and progress_token:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=p,
                total=100,
                message=msg
            )
    await send(10, "Starting X11...")
    await start_x11()
    await send(50, "Launching browser…")
    await launch_browser()
    await send(100, "ready")
```

Those messages are displayed live on hud.ai alongside resource graphs – perfect feedback while you wait.

### 4. Live telemetry (`telemetry://live`) (Optional)

Expose a resource named `telemetry://live` exactly like in `environments/browser/src/hud_controller/server.py` to return live url to be displayed on hud.ai.

Once all of the above works you can unleash *hundreds* of concurrent agents on your new environment.

---

## Phase 5 – Hot-Reload Development

For rapid local development, run the controller and environment servers separately. This enables instant code updates without Docker rebuilds.

### Development Setup

You'll need **two terminal windows** for local development:

#### Terminal 1: MCP Server
```bash
cd environments/my-environment/server
hud dev                  # Auto-detects and runs with hot-reload

# Optional flags:
hud dev --inspector      # Launch MCP Inspector
hud dev --interactive    # Launch interactive testing mode
hud dev --stdio          # Use stdio transport (default: HTTP)
hud dev --watch ../shared  # Watch additional directories
```

The `hud dev` command:
- Auto-detects the MCP module in the current directory
- Watches for file changes and reloads automatically
- Runs on HTTP by default (http://localhost:8765/mcp)
- Can launch MCP Inspector for testing tools
- Can launch interactive mode for manual testing

#### Terminal 2: Environment Server (Backend)
```bash
cd environments/my-environment/environment
uvicorn server:app --reload  # Standard uvicorn with hot-reload
```

For the backend, we simply use `uvicorn` directly since it already provides excellent hot-reload capabilities.

### Development Workflow

1. Start both servers in separate terminals
2. Edit code in either `server/` or `environment/` - changes reload automatically
3. Test changes immediately without rebuilding Docker images
4. Use MCP Inspector or interactive mode to test tools
5. When ready, build the complete Docker image: `hud build`

### Quick Cursor Setup

Add to `.cursor/mcp.json` (or use the deeplink from `hud dev` output):

```json
{
  "mcpServers": {
    "my-environment-dev": {
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

**Note**: Make sure both MCP server and environment backend are running when using with Cursor or agents.

### Process Separation for Stateful Environments

**Important Architecture Pattern**: For environments that maintain state (browsers, databases, running applications), you should separate the MCP server process from the actual environment process. This separation is critical for effective hot-reload development.

#### Why Process Separation?

When `hud dev` restarts your MCP server for code changes, you don't want to lose:
- Open browser windows and navigation state
- Database connections and data
- Running application state
- X11/VNC sessions
- Any expensive initialization

#### Architecture Pattern

```
┌─────────────────┐     ┌──────────────────────┐
│   MCP Server    │────▶│ Environment Process  │
│  (Restartable)  │     │    (Persistent)      │
└─────────────────┘     └──────────────────────┘
       ▲                         │
       │                         │
       └─── Communication ────────┘
          (Socket, API, gRPC)
```

#### Implementation Example

1. **Create a Context Server** (`context_server.py`):
```python
from hud.server.context import run_context_server

class PersistentEnvironmentContext:
    def __init__(self):
        self.state = {}
        self.resources = None
    
    def startup(self):
        # One-time expensive initialization
        self.resources = initialize_expensive_resources()
    
    def get_state(self):
        return self.state

if __name__ == "__main__":
    context = PersistentEnvironmentContext()
    context.startup()
    # Run on Unix socket
    asyncio.run(run_context_server(context, "/tmp/my_env_ctx.sock"))
```

2. **Connect from MCP Server** (`server.py`):
```python
from hud.server.context import attach_context

@mcp.initialize
async def initialize_environment(ctx):
    # Connect to persistent context
    persistent_ctx = attach_context("/tmp/my_env_ctx.sock")
    
    # Use existing state without reinitializing
    state = persistent_ctx.get_state()
    resources = persistent_ctx.get_resources()
```

3. **Update Dockerfile** to run both processes:
```dockerfile
# Start context server in background
CMD ["sh", "-c", "python -m hud_controller.context_server & python -m hud_controller.server"]
```

#### Communication Options

- **Unix Sockets** (recommended for local): Fast, simple, no network overhead
- **TCP/HTTP API**: Good for distributed systems
- **gRPC**: Type-safe, efficient for complex APIs
- **Shared Memory**: Ultra-fast for large data

See the `browser` environment for a complete production example of this pattern.

### 4. Cursor rules – paste this once

Inside `.cursor/rules/mcp_environment_iteration.mdc` add (or verify) the following so the agent always knows the expected iteration loop:

```mdc
---
description: Improve an MCP environment
alwaysApply: false
---
Setup
1. Make sure the user has set up the mcp config for the environment by seeing if you have access to the tools by the given name (i.e. my-environment-dev), and make sure the title is in dev mode. If not, ask the user to make a dev version!
2. Make sure you can find the source folder for this environment. Explore its contents and README.
3. Clarify the objectives and ask follow up questions on the initial query to determine precise implementation details.

Iteration
1. Use the exposed tools by the environment to interact with it. This means navigating around with a computer, editing, launching commands, whatever means accessible to you. If there are any exposed resources, try to access them to determine the structure of the calls.
2. Based on the objectives, test and verify the functionality of different tools and parts of the environment. If any tool call responds with an error, note it down. If any interaction with the environment is wrong, unexpected, incomplete, or parts of the environment are not developed fully, note it down. If any new problem sets up wrong or evaluation does not match the expected outcome, note it down. All of these inconsistencies you should note down in your TODOs.
3. Then, based on the TODOs, view the source folder and find the places where those errors would occur. Think about the system and how to fix it. Then fix it.
4. After you've fixed your TODO items, go back to step 2 and test them. Test through all of your available tools, and use feedback (such as screenshots) to determine your progress. If they now work as expected, mark them as complete. If not, continue the loop from step 2. Be extremely careful, scrupolous and attentive to all details. Never assume something is working unless you've tested it fully for all of its edge cases.
5. The only time you can exit this iteration loop is if you're adding if there is no feasible way to create input conditions to test something. In this case, ask the user for help and recap your progress. If you're simply changing tools, changing code, and still have more realistic TODOs, the restart_server tool automatically refreshes the environment and you should continue working. In *all* other cases, you must continue this iteration loop until you can come up with no more TODOs. You must not halt.```

### 5. Prompt the agent

```txt
Context: In the my-environment folder, I have a browser app environment. I've built a tool to interact with it called my-environment-dev.
Interaction: There are multiple tools to setup and evaluate the environment. There are also interaction tools for you to be able to move around it, and a screenshot tool to see the state. Use all of the available tools.
Objective: Please test if all setup, evaluation functions are working. This means you should come up with new problem definitions to test all functionality on. Be creative in how you pick edge cases to test on.
Rules: @mcp_environment_iteration.mdc
```

---

## Phase 6 – Optional Polish & Extensions

### Deeper dive into registries

An environment often needs *structured knowledge* about tasks, evaluation logic, or problem definitions.  The browser examples keep these in three explicit registries:

| Registry | Purpose | Example resource URI |
|----------|---------|----------------------|
| **Setup** | How to seed the environment before the agent starts | `setup://registry` & `setup://{env}` |
| **Evaluators** | Functions that decide success & reward | `evaluators://registry` |
| **Problems** | Bundled benchmarks / tasks with their own setup & evaluate pairs | `problems://registry` |

Each registry is just a dictionary mapping a *name* to a *class*.  Use a **decorator** to register classes:

```python
from .registry import setup, evaluator, problem

@setup("todo_seed")
class TodoSeed:
    ...

@evaluator("todo_completed")
class TodoCompleted:
    ...

@problem("todo_basic", description="Complete two todo items", difficulty="easy")
class TodoBasic:
    def get_setup(self):
        return {"name": "todo_seed", "arguments": {"num_items": 5}}
    def get_evaluation(self):
        return {"name": "todo_completed", "arguments": {"expected_count": 2}}
```

Decorators keep registration *next to the implementation* and avoid manual bookkeeping.  The server simply exposes the combined metadata through an MCP **resource**.  Follow `environments/browser/src/hud_controller/problems/registry.py` as a template and expose the JSON with `@mcp.resource("problems://registry")`.

### Other finishing touches

* **Performance** – lazy-load heavy resources, pool DB connections, cache expensive calls.
* **Security** – sandbox untrusted code, keep secrets in env vars, audit-log every tool call.
* **Creative ideas** – API simulators, network test-beds, game worlds… if it fits in Docker it can be an MCP environment.

---

## Contributing to Existing Environments

When improving existing environments, follow these guidelines:

### 1. Understanding the Environment

Before making changes:
- Read the environment's README and any documentation
- Run `hud debug <image>` to test the environment
- Run `hud analyze <image>` (after debug passes phase 3) to explore capabilities
- Explore the folder structure and identify key components
- Test existing setup/evaluate functions to understand behavior

### 2. Making Improvements

**Adding New Setup Functions**
```python
# In setup/my_new_setup.py
from . import setup
from hud.tools import BaseSetup, TextContent

@setup("my_new_setup", description="Clear description of what this does")
class MyNewSetup(BaseSetup):
    async def __call__(self, context, param1: str, param2: int = 10) -> TextContent:
        # Implementation
        return TextContent(...)
```

**Adding New Evaluators**
```python
# In evaluate/my_evaluator.py
from . import evaluator
from hud.tools import BaseEvaluator, EvaluationResult

@evaluator("my_check", description="What this evaluates")
class MyCheckEvaluator(BaseEvaluator):
    async def __call__(self, context, threshold: float) -> EvaluationResult:
        score = await context.calculate_score()
        return {
            "reward": min(score / 100, 1.0),
            "done": score >= threshold,
            "info": {"score": score, "threshold": threshold}
        }
```

### 3. Testing Your Changes

**Use `hud dev` for Hot-Reload Development**
```bash
# Navigate to the environment directory
cd environments/my-environment

# Start development server with hot-reload
hud dev --build

# In another terminal, test your changes
hud analyze hud-my-environment:dev

# Or use interactive mode to test tools directly
hud dev --build --interactive
```

The `hud dev` command automatically:
- Mounts your `src/` directory for live code updates
- Handles container lifecycle and restarts
- Provides an HTTP endpoint for testing
- Shows logs for debugging

## Testing Your Environment

Once your environment is working, create comprehensive tests to ensure it stays that way:

### Creating Test Files

Each environment should have a test file following this pattern:
- `environments/<env_name>/test_<env_name>_mcp.py`

The test file should include:
1. **Docker Build Test**: Ensure the image builds successfully
2. **MCP Initialization Tests**: Verify phases 1-3 using `hud debug`
3. **Tool-Specific Tests**: Test your environment's unique tools
4. **Integration Tests**: Test complete workflows

Example test structure:
```python
class TestMyEnvironment:
    IMAGE_NAME = "my-environment-test:latest"
    
    @classmethod
    def setup_class(cls):
        """Build Docker image before tests"""
        # Build the image
    
    def test_phase1_basic_startup(self):
        """Test container starts"""
    
    @pytest.mark.asyncio
    async def test_phase2_3_mcp_initialize_and_tools(self):
        """Test MCP init and tool discovery"""
    
    @pytest.mark.asyncio
    async def test_environment_specific_tools(self):
        """Test your custom tools"""
```

### Running Tests

You can run tests directly with pytest:

```bash
# Run all tests for an environment
cd environments/text_2048
pytest test_text_2048_mcp.py -v
```

### Test Dependencies

Add pytest to your environment's `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-asyncio>=0.20"]
```

## Summary

1. Start with a *plain* Dockerfile – verify it runs.  
2. Add a minimal FastMCP server – verify with stdio, Inspector, Docker.  
3. Implement tools – verify discovery + execution.  
4. Run the same image remotely – verify telemetry.  
5. Automate the loop with cursor-mcp.  
6. **Write comprehensive tests** – ensure reliability.
7. Polish and extend as inspiration strikes.

Happy building – and remember: **stderr is your friend, stdout belongs to MCP.** 🚀
