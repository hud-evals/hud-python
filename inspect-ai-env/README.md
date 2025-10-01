# Inspect AI Evaluations with Hud

This environment enables running [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evaluations using Hud's agent orchestration framework.

## Architecture

The system properly separates concerns between orchestration and sandbox execution:

```
Hud (Orchestration Layer)
 ├─ Loads inspect_ai Task definitions
 ├─ Converts samples to Hud tasks
 ├─ Runs agent for each sample
 └─ Calls evaluate tool for scoring
    ↓
MCP Controller (Tool Interface)
 ├─ setup - Initialize sandbox
 ├─ exec - Execute commands
 ├─ write_file - Write files
 ├─ read_file - Read files
 ├─ list_files - List directory
 └─ evaluate - Run scorer
    ↓
Docker Container (Sandbox Environment)
 └─ Provides isolated execution environment
    └─ HTTP endpoints for file/exec operations
```

**Key Principle**: The Docker container is **only** a sandbox. Hud handles all eval orchestration.

## Quick Start

### 1. Prepare Dataset

Convert an inspect_ai eval to Hud task format:

```bash
# Using environment variable
export TARGET_EVAL=mbpp
uv run python prepare_dataset.py --limit 5

# Or specify directly
uv run python prepare_dataset.py --eval mbpp --limit 5

# For custom evals
uv run python prepare_dataset.py --eval custom_evals.example_eval:example_eval
```

This creates `samples.jsonl` with Hud-formatted tasks.

### 2. Start Sandbox

```bash
hud dev --build
```

This starts the Docker container with:
- Sandbox server on port 8000 (HTTP)
- MCP controller exposing tools to agents

### 3. Run Evaluation

```bash
# Run with Claude
hud eval samples.jsonl --agent claude

# Run with other agents
hud eval samples.jsonl --agent gpt-4o
```

## How It Works

### Dataset Preparation (`prepare_dataset.py`)

1. **Load Task**: Uses `inspect_loader.py` to import and call the eval's task function
2. **Analyze Requirements**: Determines what sandbox tools are needed (exec, file ops, git, etc.)
3. **Convert Samples**: Uses `task_converter.py` to convert each Sample to Hud task format
4. **Apply Prompt Template**: Extracts and applies the solver's prompt template
5. **Save Tasks**: Outputs JSONL file with one task per line

### During Evaluation

1. **Hud** reads a task and gives the prompt to the agent
2. **Agent** uses MCP tools (`exec`, `write_file`, etc.) to work in the sandbox
3. **Controller** (`controller/tools.py`) forwards tool calls to sandbox server
4. **Sandbox** (`environment/server.py`) executes operations in isolated environment
5. **Evaluate Tool** runs the inspect_ai scorer to grade the output
6. **Hud** receives the reward and moves to next sample

## File Structure

```
inspect-ai-env/
├── prepare_dataset.py      # Convert inspect evals to Hud tasks
├── inspect_loader.py        # Load and analyze inspect tasks
├── task_converter.py        # Convert Task → Hud format
│
├── controller/
│   ├── __init__.py         # MCP server setup
│   ├── __main__.py         # Entry point
│   ├── hooks.py            # Lifecycle hooks
│   └── tools.py            # MCP tools (setup, exec, evaluate, etc.)
│
├── environment/
│   └── server.py           # Sandbox HTTP server
│
├── inspect_evals/          # Downloaded inspect evals
├── custom_evals/           # Your custom evals
└── Dockerfile              # Sandbox container
```

## Adding New Evals

### Official Inspect Evals

```bash
# Just specify the eval name
uv run python prepare_dataset.py --eval swe_bench --limit 5
```

The system automatically:
- Loads the eval from `inspect_evals`
- Analyzes required tools
- Converts to Hud format

### Custom Evals

1. Create your eval following inspect_ai patterns:

```python
# custom_evals/my_eval/my_eval.py
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import match

@task
def my_eval():
    return Task(
        dataset=[
            Sample(input="Your prompt", target="Expected answer", id="1"),
        ],
        solver=generate(),
        scorer=match(),
    )
```

2. Prepare dataset:

```bash
uv run python prepare_dataset.py --eval custom_evals.my_eval:my_eval
```

## Eval-Specific Tools

Different evals need different sandbox capabilities:

- **MBPP** (Python coding): Needs `exec` for running Python code
- **SWE-Bench** (bug fixing): Needs `exec`, `write_file`, `read_file`, git operations
- **Web evals**: Need browser automation tools

The system automatically detects requirements by analyzing the eval's scorer and solver.

## Configuration

### Task Parameters

Pass parameters to the task function:

```bash
uv run python prepare_dataset.py --eval mbpp \
    --task-params '{"temperature": 0.0}'
```

### MCP Configuration

Customize sandbox connection in `mcp_config` (default is local Docker):

```json
{
  "local": {
    "url": "http://localhost:8765/mcp"
  }
}
```

## Known Issues

### Dataset Preparation Dependencies

**Issue**: Some inspect_ai evals require heavy dependencies during dataset loading (e.g., `hydra-core`, `jinja2`, `torch`, `tiktoken`, `nltk`, `lxml`). Since `prepare_dataset.py` runs on the **host** (not in Docker), these dependencies would need to be installed in your host Python environment.

**Why This Happens**: Some evals do complex processing during dataset loading:
- `agent_bench`: Generates Docker compose files per sample using jinja2 templates
- `abstention_bench`: Uses hydra/omegaconf to load YAML configurations
- `bold`: Loads PyTorch models during dataset initialization
- `infinite_bench`: Uses tiktoken for token counting in samples

**Solution (Planned)**: Hud will pre-process these complex evals in an environment with all dependencies, then upload the prepared datasets to HuggingFace. This will allow dataset loading without heavyweight dependencies.

**Current Workarounds**:

1. **Skip complex evals**: Many evals work fine without extra deps (bbh, mmlu, mbpp, math, etc.)

2. **Install deps on host** (temporary):
   ```bash
   uv pip install hydra-core jinja2 torch tiktoken nltk lxml
   ```

3. **Use pre-processed datasets** (when available): Coming soon - simplified HF datasets for complex evals

### Deprecated HuggingFace Dataset Scripts

Some evals use custom dataset loading scripts that are deprecated in newer HuggingFace `datasets` versions:
- `apps`, `bbq`, `medqa`: Error "Dataset scripts are no longer supported"

These will be migrated to modern HuggingFace dataset formats.

### Gated Datasets

Some datasets require manual access approval:
- `gaia`, `hle`, `mask`, `lingoly`: Visit the dataset page on HuggingFace to request access

## Troubleshooting

### Import Errors

If the eval can't be found:
- Ensure inspect_evals is installed: `uv pip install inspect_ai inspect_evals`
- Check the eval name spelling
- For custom evals, ensure the module path is correct

### Sandbox Connection Failed

If agent can't connect to sandbox:
- Check `hud dev --build` is running
- Verify port 8765 is accessible
- Check Docker container logs

### Scorer Errors

If evaluation fails:
- Check the scorer has access to required tools
- Verify the agent's output format matches expectations
- Look at controller logs in Docker container

## Advanced Usage

### Limit Samples for Testing

```bash
uv run python prepare_dataset.py --eval mbpp --limit 10
```

### Download Eval Assets

Some evals require downloading datasets first:

```bash
uv run python prepare_dataset.py --eval mbpp --download
```

### Inspect Capabilities

Check what tools the sandbox provides:

```bash
curl http://localhost:8000/capabilities
```

## Differences from Native Inspect AI

This integration maintains compatibility with inspect_ai evals while adapting them for Hud:

1. **Orchestration**: Hud handles the eval loop, not inspect_ai's `eval()` function
2. **Model Interface**: Agents use MCP tools instead of inspect_ai's ModelAPI
3. **Sandbox**: Docker container provides sandbox, not inspect_ai's built-in sandbox
4. **Scoring**: Scorer still uses inspect_ai code but runs in controller context

## Contributing

To add support for new eval types:

1. Test with `prepare_dataset.py` to see what tools are detected
2. If needed, add tool detection logic in `inspect_loader.py`
3. Implement new tools in `controller/tools.py` and `environment/server.py`
4. Update this README with examples

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

## References

- [Inspect AI Documentation](https://inspect.ai-safety-institute.org.uk/)
- [Hud Documentation](https://docs.hud.so/)
- [inspect_evals Repository](https://github.com/UKGovernmentBEIS/inspect_evals)
