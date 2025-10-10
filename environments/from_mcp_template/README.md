# test-test

## Environment design pattern
- Controller (Think of this as a frontend in web development)
  - Creates the UX and manages the lifecycle of an app (in this case for an agent)
  - Define `mcp = MCPServer()` and register `@mcp.tool` as tools the agent can interact with
- Environment (Think of this as a backend in web development)
  - Owns all long‑lived states of the environment and exposes the environment data structure
  - Expose simple HTTP endpoints (`/health`, `/act`, `/reset`, `/state`)

IMPORTANT: Make sure all logs are going to stderr instead of stdio, which is reserved for MCP communication

### Testing your environment
```bash
# 1. Configure your API keys (optional - only needed for evaluation)
# Edit .env file to add your HUD_API_KEY and ANTHROPIC_API_KEY

# 2. Start the environment (optional: with --inspector or --interactive)
hud dev --build --interactive

# 3. Choose your preferred way to test:

# Option A: Run the task with Claude (requires ANTHROPIC_API_KEY)
hud eval tasks.json --agent claude

# Option B: Interactive notebook test_env.ipynb (great for learning!)

# Option C: Simple Python script (runs all tasks from tasks.json)
python test_task.py
```

## Iterating on your environment
This is usually the process for making any environment better:
```bash
# 1. Start the environment and interact with it directly (or give MCP server to an agent):
hud dev --build --interactive

# 2. If the environment cannot start or fails inexplicably:
hud debug test_env:dev # Or your env name that appears when you run hud dev
# After fixing the error, go back to 1.

# 3. When the environment is in a stable state:
hud build
hud push # Requires docker login

# 4. As soon as it's pushed to the newest version, make sure tasks have it updated and run:
hud rl
# This is a good test to see if your environment and tasks are high quality!

## Layout
```
controller/
  __init__.py   # mcp + shared HTTP client
  __main__.py   # python -m controller → mcp.run()
  hooks.py      # @mcp.initialize / @mcp.shutdown
  tools.py      # @mcp.tool act / setup / evaluate

./environment
  ├── __init__.py
  └── server.py       # FastAPI app: /health, /act, /reset, /state
```

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
# hud.so/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.so/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.so/evaluate-agents/leaderboards)

