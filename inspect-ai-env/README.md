# Inspect AI + HUD Integration

Run any [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) benchmark through your HUD agent with full control over all LLM interactions.

## What This Does

- **Runs 60+ evaluations** (MBPP, SWE-bench, GPQA, HumanEval, etc.) using their native solvers and scorers
- **Routes all LLM calls through your HUD agent** instead of calling APIs directly
- **Provides MCP tools** (`setup`, `evaluate`) to control evaluations
- **Maintains compatibility** with inspect_ai's official evaluation logic

## Quick Start

### 1. Build the Docker Environment

```bash
cd hud-python/inspect-ai-env
hud dev --build
```

This installs `inspect-ai` and `inspect-evals` in the Docker container.

### 2. Run an Evaluation

```python
from hud.clients import MCPClient
import asyncio

async def run_eval():
    client = MCPClient(mcp_config={
        "inspect_ai_env": {"url": "http://localhost:8765/mcp"}
    })
    await client.initialize()

    # Setup environment
    await client.call_tool(name="setup")

    # Run MBPP with 3 samples
    result = await client.call_tool(
        name="evaluate",
        arguments={
            "eval_name": "mbpp",
            "task_params": {"temperature": 0.5},
            "limit": 3
        }
    )

    print(result.content)
    await client.shutdown()

asyncio.run(run_eval())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Host Machine                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Your Agent Server (port 9000)                        │ │
│  │  - Receives generate() requests via HTTP              │ │
│  │  - Calls actual LLM API (Claude, GPT-4, etc.)        │ │
│  │  - Returns responses                                   │ │
│  └──────────────────────────▲────────────────────────────┘ │
│                              │                              │
│                              │ HTTP POST (AGENT_CALLBACK_URL)│
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────┐
│          Docker Container    │                              │
│                              │                              │
│  ┌───────────────────────────┴──────────────────────────┐  │
│  │  Environment Server (port 8000)                      │  │
│  │                                                       │  │
│  │  @app.post("/model/generate")                        │  │
│  │  - Reads AGENT_CALLBACK_URL env var                  │  │
│  │  - Forwards to host agent server                     │  │
│  │  - Returns response to HUDAgentModel                 │  │
│  └──────────────────────────▲───────────────────────────┘  │
│                              │ HTTP POST                    │
│  ┌───────────────────────────┴──────────────────────────┐  │
│  │  HUDAgentModel (custom ModelAPI)                     │  │
│  │  - Intercepts all generate() calls from inspect_ai   │  │
│  │  - Routes to environment server                      │  │
│  └──────────────────────────▲───────────────────────────┘  │
│                              │ generate() call              │
│  ┌───────────────────────────┴──────────────────────────┐  │
│  │  Inspect AI Evaluation                                │  │
│  │  @app.post("/evaluate")                               │  │
│  │  - Loads eval from inspect_evals                      │  │
│  │  - Runs solver (calls generate() via HUDAgentModel)  │  │
│  │  - Runs scorer (validates responses)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ▲                              │
│                              │ HTTP POST                    │
│  ┌───────────────────────────┴──────────────────────────┐  │
│  │  MCP Controller                                       │  │
│  │  @mcp.tool("evaluate")                                │  │
│  │  - Forwards to environment server                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ▲                              │
└──────────────────────────────┼──────────────────────────────┘
                               │ MCP protocol
┌──────────────────────────────┼──────────────────────────────┐
│                       Host Machine                          │
│                                                             │
│  MCPClient.call_tool("evaluate", args=...)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### MCP Tools (controller/tools.py)

**`setup(eval_name)`** - Initialize the environment
```python
# Basic setup (no extra installs)
await client.call_tool(name="setup")

# Setup with automatic eval-specific dependency installation
await client.call_tool(
    name="setup",
    arguments={"eval_name": "swe_bench"}
)
```

**Note**: When you provide an `eval_name`, the setup tool automatically attempts to install
eval-specific dependencies using `uv pip install inspect_evals[eval_name]`. This handles evals that
need extra packages:
- `swe_bench` → `swebench>=3.0.15`, `docker`
- `mathematics` → `sympy`, `antlr4-python3-runtime==4.13.2`
- `mle_bench` → `mlebench`, `docker`
- etc.

The installation is done with try/except, so evals without extra dependencies (like `mbpp`)
won't cause errors.

**`evaluate(eval_name, task_params, limit)`** - Run full evaluation
```python
await client.call_tool(
    name="evaluate",
    arguments={
        "eval_name": "mbpp",
        "task_params": {"temperature": 0.5},
        "limit": 5
    }
)
```

### HUDAgentModel (environment/hud_model.py)

Custom `ModelAPI` provider that intercepts inspect_ai's model calls:

```python
@modelapi(name="hud")
class HUDAgentModel(ModelAPI):
    async def generate(self, input, tools, config):
        # Intercepts generate() calls from inspect_ai
        # Routes to /model/generate endpoint
        response = await http_client.post(
            "http://localhost:8000/model/generate",
            json={...}
        )
        return ModelOutput.from_content(response["content"])
```

### Environment Server (environment/server.py)

**`POST /evaluate`** - Runs inspect_ai evaluation with `model="hud/agent"`

**`POST /model/generate`** - Receives model calls, should route to your agent
```python
@app.post("/model/generate")
async def model_generate(request: ModelGenerateRequest):
    # TODO: Implement routing to your external HUD agent
    # For now returns mock response
    return {"content": "..."}
```

## Supported Evaluations

All 60+ inspect_evals work automatically:

**Code Generation:**
- mbpp, humaneval, apps, bigcodebench, class_eval, ds1000

**Software Engineering:**
- swe_bench, swe_bench_verified

**Math & Science:**
- gsm8k, math, gpqa, aime

**Reasoning:**
- arc, hellaswag, mmlu, bbh, commonsense_qa

**Agents:**
- gaia, assistant_bench

**Security:**
- cybench, cybermetric, cyberseceval_2

See `inspect_evals/` for the full list.

## Configuration

### Eval Parameters

Each eval accepts different parameters passed via `task_params`:

**MBPP:**
```python
task_params = {"temperature": 0.5}
```

**SWE-bench:**
```python
task_params = {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "instance_ids": ["django__django-12184"],
    "max_messages": 30,
    "build_docker_images": False
}
```

**GPQA:**
```python
task_params = {"dataset": "gpqa_diamond"}
```

See eval source in `inspect_evals/src/inspect_evals/{eval_name}/` for all parameters.

### Limiting Samples

Use the `limit` parameter to test with fewer samples:

```python
arguments={
    "eval_name": "mbpp",
    "limit": 3  # Only run 3 samples
}
```

## Connecting Your Agent

The system routes all LLM calls from inspect_ai to your external agent via HTTP callback.

### Setup

1. **Create an agent server on your host machine:**

```python
# host_agent_server.py
from fastapi import FastAPI
from anthropic import Anthropic

app = FastAPI()
client = Anthropic()

@app.post("/generate")
async def generate(request: dict):
    messages = request["messages"]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        max_tokens=4096
    )

    return {
        "content": response.content[0].text,
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn"
    }

# Run on host: uvicorn host_agent_server:app --host 0.0.0.0 --port 9000
```

2. **Set the callback URL environment variable:**

```bash
# Add to .env file
AGENT_CALLBACK_URL=http://host.docker.internal:9000/generate
```

Or set it when running:

```bash
export AGENT_CALLBACK_URL=http://host.docker.internal:9000/generate
hud dev --build
```

3. **That's it!** The system will now route all model calls to your agent.

### How It Works

1. Inspect AI calls `generate()`
2. HUDAgentModel intercepts and forwards to `/model/generate`
3. Environment server reads `AGENT_CALLBACK_URL` and forwards request
4. Your host agent receives the request and calls the actual LLM API
5. Response flows back through the chain

### Without Agent Connection

If `AGENT_CALLBACK_URL` is not set, the system returns mock responses. This is useful for testing the pipeline without an actual agent.

## How It Works

### 1. When You Call `evaluate`

```python
await client.call_tool(name="evaluate", arguments={"eval_name": "mbpp", "limit": 3})
```

### 2. Environment Server Runs Inspect AI

```python
# Registers HUD model provider
from environment.hud_model import HUDAgentModel

# Runs eval with custom model
logs = await inspect_eval(
    task,
    model="hud/agent",  # Uses HUDAgentModel instead of OpenAI/Anthropic
    log_dir="logs"
)
```

### 3. Solver Needs LLM Response

When the eval's solver calls `generate()`:

```python
# Inside MBPP solver
output = await generate(input="Write a Python function...")
```

### 4. HUDAgentModel Intercepts

```python
# In environment/hud_model.py
async def generate(self, input, tools, config):
    # Routes to environment server
    response = await http_client.post(
        "http://localhost:8000/model/generate",
        json={"messages": [...], "tools": [...]}
    )
    return ModelOutput.from_content(response["content"])
```

### 5. Environment Server Routes to Your Agent

```python
@app.post("/model/generate")
async def model_generate(request):
    # TODO: Call your external agent here
    # For now: mock response
    return {"content": "def solution(): pass"}
```

### 6. Response Flows Back

The response flows back through the chain:
```
Your Agent → Environment Server → HUDAgentModel → Inspect AI Solver → Scorer
```

### 7. Scorer Validates

The eval's native scorer validates the response:
```python
# In MBPP scorer
result = await sandbox().exec(["python", "-c", generated_code])
score = CORRECT if result.success else INCORRECT
```

## Benefits

✅ **Full Control**: Intercept every LLM call
✅ **Monitoring**: Log all prompts and responses
✅ **Cost Tracking**: Monitor token usage per eval
✅ **Custom Logic**: Add reasoning, RAG, tool use before LLM
✅ **Model Switching**: Easily switch between models
✅ **Official Scoring**: Uses each eval's native scorer (guaranteed correct)

## Files Overview

```
inspect-ai-env/
├── controller/
│   ├── __init__.py         # MCP server setup
│   ├── tools.py            # MCP tools (setup, evaluate, process_sample)
│   └── hooks.py            # MCP hooks
├── environment/
│   ├── server.py           # FastAPI server (evaluate, model_generate endpoints)
│   └── hud_model.py        # Custom ModelAPI for routing
├── inspect_evals/          # Downloaded evals (via download-eval.sh)
│   └── mbpp/
├── docker_pyproject.toml   # Dependencies (inspect-ai, inspect-evals)
├── Dockerfile              # Container setup
├── download-eval.sh        # Script to download evals
├── tasks.json              # Task configuration
└── README.md               # This file
```

## Development Workflow

### 1. Add New Eval

```bash
# Download the eval
TARGET_EVAL=swe_bench ./download-eval.sh

# Or add to Dockerfile
ENV TARGET_EVAL=swe_bench
RUN ./download-eval.sh
```

### 2. Test Evaluation

```python
result = await client.call_tool(
    name="evaluate",
    arguments={
        "eval_name": "swe_bench",
        "limit": 1  # Test with 1 sample first
    }
)
```

### 3. Implement Agent Routing

Update `environment/server.py:model_generate()` to call your agent.

### 4. Scale Up

Remove `limit` parameter to run full evaluation.

## Troubleshooting

### "Eval not found"
The eval needs to be downloaded. Add it to `download-eval.sh` or rebuild the image.

### "Model not found"
Ensure HUDAgentModel is imported in `environment/server.py`.

### Mock Responses
If you're getting mock responses, implement the agent routing in `/model/generate`.

### Timeout Errors
Increase timeout in `controller/tools.py`:
```python
timeout=600.0,  # 10 minutes
```

## Next Steps

1. **Implement Agent Routing**: Update `/model/generate` in `environment/server.py`
2. **Test with Small Eval**: Run MBPP with `limit=1`
3. **Add Logging**: Track all model calls
4. **Scale Up**: Run full evaluations
5. **Monitor Costs**: Track token usage through your agent

## Using Custom Evals

You can run your own custom evals that are compatible with inspect_ai format but not in the official inspect_evals package.

### Quick Start: Run the Example

We include an example custom eval to help you get started:

```bash
# Build with custom_evals directory mounted (it's already in the repo)
cd hud-python/inspect-ai-env
hud dev --build

# Run the example eval
python run_task.py custom_evals.example_eval --limit 2

# Or with parameters
python run_task.py custom_evals.example_eval:example_eval_with_params \
    --task-params '{"difficulty": "medium"}'
```

The example eval is in `custom_evals/example_eval/example_eval.py` - use it as a template!

### Directory Structure

Mount your custom eval code into the Docker container at `/app/custom_evals/`:

```
custom_evals/
├── __init__.py
└── my_eval/
    ├── __init__.py
    └── my_eval.py  # Contains your task function
```

### Task Function Format

Your custom eval should follow the inspect_ai Task format:

```python
# custom_evals/my_eval/my_eval.py
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import match

@task
def my_eval():
    """My custom evaluation task."""
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        solver=[
            system_message("You are a helpful assistant."),
            generate()
        ],
        scorer=match()
    )
```

### Mounting Custom Evals

Update your `docker-compose.yml` or use volume mounts:

```yaml
# docker-compose.yml
services:
  inspect-ai-env:
    volumes:
      - ./my_custom_evals:/app/custom_evals
```

Or with `hud dev`:

```bash
# Add volume mount to your HUD configuration
hud dev --build -v ./my_custom_evals:/app/custom_evals
```

### Running Custom Evals

Use the module path as the eval_name:

```python
from hud.clients import MCPClient

client = MCPClient(mcp_config={
    "inspect_ai_env": {"url": "http://localhost:8765/mcp"}
})
await client.initialize()

# Setup with custom eval name
await client.call_tool(name="setup", arguments={"eval_name": "custom_evals.my_eval"})

# Run evaluation
result = await client.call_tool(
    name="evaluate",
    arguments={
        "eval_name": "custom_evals.my_eval",  # Module path
        "limit": 2
    }
)
```

### Advanced: Explicit Function Names

If your task function has a different name than the module:

```python
# custom_evals/my_eval/my_eval.py
@task
def custom_task_function():  # Different from module name
    return Task(...)
```

Specify it explicitly:

```python
result = await client.call_tool(
    name="evaluate",
    arguments={
        "eval_name": "custom_evals.my_eval:custom_task_function",  # module:function
        "limit": 2
    }
)
```

### Custom Dataset Files

You can also load datasets from files in your custom eval:

```python
from inspect_ai.dataset import json_dataset

@task
def my_eval(dataset_path: str = "dataset.jsonl"):
    return Task(
        dataset=json_dataset(dataset_path),
        solver=[...],
        scorer=[...]
    )
```

Mount the dataset file alongside your code:

```bash
hud dev --build \
  -v ./my_custom_evals:/app/custom_evals \
  -v ./my_datasets:/app/datasets
```

Then pass the path:

```python
result = await client.call_tool(
    name="evaluate",
    arguments={
        "eval_name": "custom_evals.my_eval",
        "task_params": {"dataset_path": "/app/datasets/my_data.jsonl"},
        "limit": 10
    }
)
```

## Additional Resources

- Inspect AI docs: https://inspect.ai-safety-institute.org.uk/
- Inspect Evals repo: https://github.com/UKGovernmentBEIS/inspect_evals
- HUD docs: https://docs.hud.so/