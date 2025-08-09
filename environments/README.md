# How to Build HUD-Compatible MCP Environments

This document is a step-by-step guide for turning *any* piece of software that can run in a Docker container into a **Model Context Protocol (MCP)** environment that the HUD SDK can evaluate or control.  We’ll move through six short phases, each with a clear checkpoint.

The official MCP lifecycle specification is an excellent companion reference – skim it now, keep it open while you work: [modelcontextprotocol.io › Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).

---

## Phase Overview

| Phase | Goal |
|-------|------|
| 1 | A Docker image that *starts* and prints to **stderr** |
| 2 | A minimal MCP server that responds to `initialize` over **stdio** |
| 3 | Working `setup`, `evaluate`, and **interaction** tools |
| 4 | Image launches remotely on the HUD platform & exposes live telemetry |
| 5 | Fast local iteration with Cursor Agent and a tiny `mcp.json` |

Take the phases one at a time; do **not** jump ahead.  Each stage's checkpoint is the foundation for the next.

💡 **Example to follow along:** The `environments/text_2048/` folder contains a complete implementation of a simple 2048 game environment. It's an excellent reference showing all phases in action with minimal complexity. Check it out as you work through each phase!

### One-command sanity check (`docker_debug.py`)

While you move through the phases it’s handy to run the **interactive checker** to make sure nothing broke:

```bash
python environments/docker_debug.py my-environment:latest
```

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
├── pyproject.toml          # Package definition with dependencies
├── README.md               # Environment documentation
└── src/
    └── my_module/          # Your Python package
        ├── __init__.py
        ├── server.py       # MCP server entry point
        ├── context.py      # Core stateful environment logic (optional)
        ├── tools/          # Interactive tools (move, click, type, etc.)
        │   ├── __init__.py
        │   └── move.py     # Example: custom tool inheriting from BaseTool
        ├── setup/          # Setup functions (modular approach)
        │   ├── __init__.py # Creates SetupTool instance & exports decorator
        │   ├── basic.py    # Basic setup functions
        │   └── advanced.py # Advanced setup functions
        └── evaluate/       # Evaluator functions (modular approach)
            ├── __init__.py # Creates EvaluateTool instance & exports decorator
            ├── checks.py   # Basic evaluation checks
            └── metrics.py  # Advanced metrics evaluators
```

This structure enables:
- Clean separation of concerns (environment logic, tools, setup, evaluation)
- Easy volume mounting for development (Phase 5)
- Standard Python packaging with `pip install -e .`
- Modular organization - each setup/evaluator in its own file for clarity

• **One Dockerfile only** – no docker-compose.  
• If you're building a GUI environment, start from `hudpython/novnc-base:latest` instead and leave VNC configuration for later phases.

Checkpoint reached?  Congratulations – move on.

👉 Quick sanity check: `python environments/docker_debug.py my-environment:latest` (verifies Phase 1 automatically)

Need inspiration?  Skim the real Dockerfiles used in the example browser environments:
• [`text_2048/Dockerfile`](./text_2048/Dockerfile)
• [`browser/Dockerfile`](./browser/Dockerfile)
They follow the exact same pattern – a single file, logs to stderr, nothing fancy.

---

## Phase 2 – Create the MCP Server

**Goal →** a Python process that:
1. Speaks MCP over **stdio**.
2. Responds correctly to the `initialize` request.
3. Logs everything to **stderr**.

The MCP lifecycle is *initialize → operate → shutdown* (see spec link above).

### Skeleton server (FastMCP)

```python
import sys
import logging
from mcp.server.fastmcp import FastMCP

# 1️⃣  Always log to stderr – stdout is reserved for JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s'
)

mcp = FastMCP("My Environment")

from hud.tools.helper import mcp_intialize_wrapper

@mcp_intialize_wrapper()
async def initialize_environment():
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
| 4 | **Run `docker_debug.py`** – `python environments/docker_debug.py my-environment:latest` | Combines the above checks & points out common mistakes |

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

👉 After wiring in the tools, confirm with `python environments/docker_debug.py my-environment:latest` – it now checks for their presence and basic execution.

1. Write **`setup`** and **`evaluate`** tools first – they are *lifecycle* tools and never shown to the LLM.
2. Register at least one **interaction** tool (`computer`, `playwright`, or your own).

### Approach 1: Simple Direct Implementation

For simple environments with just a few setup/evaluate functions:

```python
from hud.tools.helper import register_instance_tool
from hud.tools import HudComputerTool

@mcp.tool()
async def setup(config: dict) -> dict:
    ...               # prepare environment

@mcp.tool()
async def evaluate(config: dict) -> dict:
    ...               # return {"reward": <0-1>, "done": bool}

@mcp.initialize()
async def init():
    register_instance_tool(mcp, "computer", HudComputerTool())
```

### Approach 2: Registry Pattern (Recommended for Complex Environments)

For environments with multiple setup/evaluate functions, use the registry pattern with modular organization:

```python
# In setup/__init__.py
from hud.tools import SetupTool

# Create global tool instance
setup_tool = SetupTool(name="setup", title="Environment Setup")
setup = setup_tool.register  # Export decorator for convenience

# Import all setup modules to register their functions
from . import basic, advanced  # This registers all @setup decorated classes

# In setup/basic.py
from . import setup
from hud.tools import BaseSetup, SetupResult

@setup("reset", description="Reset environment to initial state")
class ResetSetup(BaseSetup):
    async def __call__(self, context, **kwargs) -> SetupResult:
        # Context is passed as first argument
        await context.reset_state()
        return {"status": "success", "message": "Reset complete"}

@setup("seed_data", description="Seed with test data")
class SeedDataSetup(BaseSetup):
    async def __call__(self, context, num_items: int = 5) -> SetupResult:
        # Type hints for parameters are preserved
        items = await context.create_items(num_items)
        return {"status": "success", "items_created": len(items)}

# In evaluate/__init__.py
from hud.tools import EvaluateTool

# Create global tool instance
evaluate_tool = EvaluateTool(name="evaluate", title="Task Evaluator")
evaluator = evaluate_tool.register  # Export decorator

# Import all evaluator modules
from . import checks, metrics

# In evaluate/checks.py
from . import evaluator
from hud.tools import BaseEvaluator, EvaluationResult

@evaluator("task_complete", description="Check if task is done")
class TaskCompleteEvaluator(BaseEvaluator):
    async def __call__(self, context, expected_count: int) -> EvaluationResult:
        # Must return dict with 'reward' (0-1) and 'done' (bool)
        completed = await context.count_completed()
        return {
            "reward": min(completed / expected_count, 1.0),
            "done": completed >= expected_count,
            "info": {"completed": completed, "expected": expected_count}
        }

# In server.py
from .setup import setup_tool
from .evaluate import evaluate_tool
from hud.tools.helper import register_instance_tool

@mcp.resource("setup://registry")
async def get_setup_registry() -> str:
    """Expose available setup functions"""
    return setup_tool.to_json()

@mcp.resource("evaluators://registry")
async def get_evaluator_registry() -> str:
    """Expose available evaluators"""
    return evaluate_tool.to_json()

@mcp_intialize_wrapper()
async def initialize_environment():
    # Initialize your environment state/context
    context = await create_environment_context()
    
    # Set context for tools (shared state)
    setup_tool.context = context
    evaluate_tool.context = context
    
    # Register tools with MCP
    register_instance_tool(mcp, setup_tool)
    register_instance_tool(mcp, evaluate_tool)
    
    # Register interaction tools
    if hasattr(context, 'custom_tool'):
        register_instance_tool(mcp, context.custom_tool)
```

This registry pattern provides:
- **Modular organization**: Each setup/evaluator in its own file or grouped logically
- **Auto-discovery**: Import modules in `__init__.py` to auto-register functions
- **Type safety**: Full type hints preserved for parameters and returns
- **Shared context**: Single context object passed to all functions

### Test workflow

1. **Inspector first** – restart the server, refresh the *Tools* tab, confirm the new tools appear.  
2. **Rebuild the image** – `docker build -t my-environment .`.  
3. **HUD SDK test** – run a short script like the one below.  GUI environments built from `hudpython/novnc-base` still expose a VNC viewer on <http://localhost:8080/vnc.html> – keep it open while testing.

```python
import asyncio
from hud.datasets import TaskConfig
from hud.mcp import ClaudeMCPAgent, MCPClient
from hud.telemetry import trace

async def main():
    # `trace` captures *everything* that happens and sends it to app.hud.so
    with trace("local_test"):
        task = TaskConfig(
            prompt="Complete the task",
            mcp_config={
                "local": {
                    "command": "docker", 
                    "args": ["run", "--rm", "-i", "my-environment:latest"]
                }
            }
            setup_tool={"name": "setup", "arguments": {"name": "todo_seed", "num_items": 5}},
            evaluate_tool={"name": "evaluate", "arguments": {"name": "todo_completed", "expected_count": 2}}
        )
        client = MCPClient(mcp_config=task.mcp_config)

        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["computer"]  # or ["move"] for text_2048
        )

        result = await agent.run(task)
        print(result)

    await client.close()

asyncio.run(main())
```

The `trace` context manager sends a full timeline of agent actions, tool calls, and rewards to app.hud.so – perfect for debugging.

See `examples/agents_tools/simple_task_example.py` and `examples/environments/gmail_local.py` for larger end-to-end demos.

---

## Phase 4 – Remote Deployment & HUD Runner

**Goal →** the exact same image runs in parallel on hundreds of instances, and exposes more telemetry so the app.hud.so can visualise the whole lifecycle.

### 1. Publish your image

Log in to Docker Hub (or any registry HUD can pull from) and push a tagged build:

```bash
docker tag my-environment yourdockerhubuser/my-environment:latest
docker push yourdockerhubuser/my-environment:latest
```

*(If you’re using a private registry, make sure the HUD worker has pull credentials.)*

### 2. Launch it remotely (gmail_remote pattern)

`examples/environments/gmail_remote.py` shows the canonical pattern – a remote MCP server entry that simply runs **the same Docker image**:

```python
from hud import settings
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

At remote scale it can take 10-30 s for heavy services to boot.  Use `mcp_intialize_wrapper()` with a *progress token* to stream status messages:

```python
from hud.tools.helper import mcp_intialize_wrapper

@mcp_intialize_wrapper()
async def initialize_environment(session=None, progress_token=None):
    async def send(p, msg):
        if session and progress_token:
            await session.send_progress_notification(progress_token, p, 100, msg)
    await send(10, "starting X11…")
    await start_x11()
    await send(50, "launching browser…")
    await launch_browser()
    await send(100, "ready")
```

Those messages are displayed live on app.hud.so alongside resource graphs – perfect feedback while you wait.

### 4. Live telemetry (`telemetry://live`) (Optional)

Expose a resource named `telemetry://live` exactly like in `environments/browser/src/hud_controller/server.py` to return live url to be displayed on app.hud.so.

Once all of the above works you can unleash *hundreds* of concurrent agents on your new environment.

---

## Phase 5 – Takeoff: Automatic environment improvement with Cursor Agent

To enable rapid development without Docker rebuilds, we can mount the dockerfile and expose the live MCP server to Cursor Agent or any other MCP client. We can combine this approach with a package like [reloaderoo](https://github.com/cameroncooke/reloaderoo) that is a proxy to allow dynamic reloading of the MCP connection, so the entire agent loop can happen asynchronously.

### Setting up Development Mode

#### 1. Build for Development

Your Dockerfile needs to copy source for the build, even though we'll mount over it:

```dockerfile
# Copy source files
COPY src/ ./src/

# Install in editable mode for development
RUN pip install -e .
```

#### 2. Build the Development Image

```bash
docker build -t my-environment:dev .
```

#### 3. Configure Cursor Agent for development with hot-reload

Add a development configuration to `.cursor/mcp.json` using [reloaderoo](https://github.com/cameroncooke/reloaderoo):

```jsonc
{
  "mcp_config": {
    // If your production config looks like this,
    "my-environment": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "my-environment:latest"]
    },
    // This is how you make the dev mode config:
    "my-environment-dev": {
      "command": "npx",
      "args": [
        "reloaderoo", "--",  // Wraps docker for hot-reload
        "docker", "run", "-i", "--rm",
        "-v", "%cd%/src:/app/src:rw",  // Windows
        // "-v", "$(pwd)/src:/app/src:rw",  // Linux/Mac
        "-e", "PYTHONPATH=/app/src",  // Required for module imports in the Phase 1 like setup
        // Add your environment variables here
        "my-environment:dev" // dev instead of latest!
      ]
    }
  }
}
```

Now you can edit code and call `restart_server` to reload without restarting the client.

2. Follow the cursor rules below: rebuild, refresh, test, reflect, repeat.
3. Keep the agent open for any messages or issues.

### 3.5. Debug MCP servers directly from Cursor (Optional)

The `docker_debug.py` utility can also run as an MCP server itself! Add this to your `.cursor/mcp.json` to debug any MCP server directly from Cursor:

```jsonc
{
  "mcpServers": {
    "mcp-debugger": {
      "command": "python",
      "args": ["/path/to/environments/docker_debug.py", "--mcp"]
    }
  }
}
```

You can use the "debug_cursor_config" tool to test another mcp server by name (like the one defined in 3.)

Example usage in Cursor:
```
Use the debug_docker_image tool to test if my-environment:dev starts correctly with max_phase=3
```

This is incredibly useful for rapid debugging without leaving your IDE!

### 4. Cursor rules – paste this once

Inside `.cursor/rules/hud_environment_iteration.mdc` add (or verify) the following so the agent always knows the expected iteration loop:

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
Rules: @hud_environment_iteration.mdc
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
        return {"function": "todo_seed", "args": {"num_items": 5}}
    def get_evaluation(self):
        return {"function": "todo_completed", "args": {"expected_count": 2}}
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
- Run `docker_debug.py` to understand current capabilities
- Explore the folder structure and identify key components
- Test existing setup/evaluate functions to understand behavior

### 2. Making Improvements

**Adding New Setup Functions**
```python
# In setup/my_new_setup.py
from . import setup
from hud.tools import BaseSetup, SetupResult

@setup("my_new_setup", description="Clear description of what this does")
class MyNewSetup(BaseSetup):
    async def __call__(self, context, param1: str, param2: int = 10) -> SetupResult:
        # Implementation
        return {"status": "success", "details": "..."}
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

**Use the Development Configuration**
```jsonc
// In .cursor/mcp.json
{
  "mcpServers": {
    "my-env-dev": {
      "command": "npx",
      "args": [
        "reloaderoo", "--",
        "docker", "run", "-i", "--rm",
        "-v", "$(pwd)/src:/app/src:rw",
        "my-environment:dev"
      ]
    }
  }
}
```

## Testing Your Environment

Once your environment is working, create comprehensive tests to ensure it stays that way:

### Creating Test Files

Each environment should have a test file following this pattern:
- `environments/<env_name>/test_<env_name>_mcp.py`

The test file should include:
1. **Docker Build Test**: Ensure the image builds successfully
2. **MCP Initialization Tests**: Verify phases 1-3 from docker_debug.py
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

Use the generic test runner:
```bash
# Run all tests for an environment
python environments/run_environment_tests.py browser

# Run specific tests
python environments/run_environment_tests.py text_2048 -k test_game_tools

# List available environments
python environments/run_environment_tests.py --list
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
