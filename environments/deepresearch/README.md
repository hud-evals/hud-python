# Deep Research Environment

Web research environment powered by Exa API for searching and fetching content.
See [docs](https://docs.hud.ai/build-environments) for the complete environment design workflow.

## Architecture

**`environment/`** - Manages Exa API integration and state
- Holds the Exa API key server-side
- Exposes HTTP endpoints `/search`, `/fetch`, `/answer`, `/evaluate` for research workflows
- Implements exponential backoff for rate limiting

**`server/`** - Wraps data in MCP tools
- Provides `search()`, `fetch()`, `answer()`, `evaluate()` tools for agents
- Agents and tasks interact only with these tools

**Why separate?** Edit tools for the agent or tasks without restarting the environment backend.

## Tools

- **`search(query: str)`** - Search the web using Exa API, returns list of results with titles and URLs
- **`fetch(url: str)`** - Fetch full content from a URL, returns summary, highlights, and text
- **`answer(final_answer: str)`** - Submit the final research answer
- **`evaluate(expected_answer: str)`** - Evaluate submitted answer against expected result

## Setup

### Requirements
- Exa API key (get one at [exa.ai](https://exa.ai))

### Environment Variables
```bash
export EXA_API_KEY="your_exa_api_key_here"
```

## Development

```bash
# Terminal 1 - Environment backend
cd environment
export EXA_API_KEY="your_key"
uv run uvicorn server:app --reload

# Terminal 2 - MCP server
cd server
uv run hud dev
```

The environment includes exponential backoff for rate limiting, so API calls will automatically retry on 429 errors.

In general, we recommend starting work on the environment backend first, then developing the MCP server to expose the right things to the agent.

For complex environments that require many dependencies, we recommend running `hud dev` in the environment root:
```bash
cd ..
export EXA_API_KEY="your_key"
hud dev
```

## Tasks & Evaluation

```bash
# Build first in the global folder with the Dockerfile (creates deepresearch:0.1.0)
hud build
```

Your `tasks.json` uses `docker run` to launch the environment:

```json
{
  "prompt": "Research and answer: What is the capital of France?",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-e", "EXA_API_KEY", "deepresearch:0.1.0"]
    }
  },
  "evaluator": {
    "tool_name": "evaluate",
    "tool_params": {
      "expected_answer": "Paris"
    }
  }
}
```

**Note:** The `-e EXA_API_KEY` flag passes your local API key to the container.

**Commands:**
```bash
# Build first
hud build

# Test task locally
export EXA_API_KEY="your_key"
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
hud eval "your-org/your-dataset" --agent claude

# View results at:
# hud.ai/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.ai/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.ai/evaluate-agents/leaderboards)

## Example Research Workflow

```python
# Agent searches for information
results = search("latest AI developments 2024")

# Agent fetches detailed content from top result
content = fetch(results[0]["url"])

# Agent submits final answer
answer("Based on research, AI developments in 2024 include...")

# Evaluate answer
result = evaluate(expected_answer="AI developments")
```
