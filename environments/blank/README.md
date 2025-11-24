# Blank Environment

Minimal starter template for building HUD environments.
See [docs](https://docs.hud.ai/build-environments) for the complete environment design workflow.

## Architecture

**`environment/`** - Produces structured data

- Owns all state (game logic, browser sessions, databases, etc.)
- Exposes HTTP endpoints `/health`, `/act`, `/reset`, `/state` that return structured information about the environment state

**`server/`** - Wraps data in MCP tools

- Calls environment endpoints to get structured data for the agent, and environment setup/evaluation
- Agents and tasks interact only with these tools!

**Why separate?** Edit tools for the agent or tasks without restarting the heavy environment backend.

## Development

```bash
# Terminal 1 - Environment backend
cd environment
uv run uvicorn server:app --reload

# Terminal 2 - MCP server
cd server
uv run hud dev
```

Uncomment the `setup` tool in `server/tools.py`, save, and watch it reload.
Visit http://localhost:8765/docs to see the new tool appear instantly.

In general, we recommend starting work on the environment backend first, then developing the MCP server to expose the right things to the agent.

For complex environments that require many dependencies, we recommend running `hud dev` in the environment root:

```bash
cd ..
hud dev
```

## Tasks & Evaluation

```bash
# Build first in the global folder with the Dockerfile (creates blank:0.1.0)
hud build
```

Your `tasks.json` uses `docker run` to launch the environment:

```json
{
  "prompt": "Your task prompt",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "blank:0.1.0"]
    }
  }
}
```

**Commands:**

```bash
# Build first
hud build

# Test task locally
hud eval tasks.json

# Push environment for remote running
hud push

# Production RL training
hud rl tasks.json  # Auto-converts docker→remote, builds & pushes if needed
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
hud eval "your-org/your-dataset" claude

# View results at:
# hud.ai/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.ai/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.ai/evaluate-agents/leaderboards)
