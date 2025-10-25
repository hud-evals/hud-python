# SEC EDGAR Rubrics Environment

SEC filing research environment powered by the SEC EDGAR database for accessing company filings and financial data, with rubric-based evaluation for structured grading.
See [docs](https://docs.hud.so/build-environments) for the complete environment design workflow.

## Architecture

**`environment/`** - Manages SEC EDGAR integration and state
- Uses the edgartools Python library to access SEC filing data
- Exposes HTTP endpoints `/search_company`, `/get_filings`, `/get_filing_content`, `/answer`, `/evaluate` for research workflows
- Implements exponential backoff for rate limiting

**`server/`** - Wraps data in MCP tools
- Provides `search_company()`, `get_filings()`, `get_recent_filings()`, `get_filing_content()`, `get_filing_content_by_accession()`, `answer()`, `evaluate()` tools for agents
- Agents and tasks interact only with these tools

**Why separate?** Edit tools for the agent or tasks without restarting the environment backend.

## Tools

- **`search_company(query: str)`** - Search for a company by ticker symbol or name. Returns company information including ticker, name, and CIK.
- **`get_filings(ticker: str, form_type: Optional[str], limit: int)`** - Get recent SEC filings for a company. Can filter by form type (e.g., "10-K", "10-Q", "8-K") and limit the number of results.
- **`get_recent_filings(identifier?: str, form_type?: str, limit?: int)`** - Global feed or company-specific recent filings. `identifier` can be ticker or CIK. If omitted, returns global recent.
- **`get_filing_content(filing_url: str)`** - Fetch the full text content of a specific SEC filing from its URL.
- **`get_filing_content_by_accession(identifier: str, accession_number: str)`** - Fetch filing content precisely using ticker/CIK and accession number (avoids URL parsing issues).
- **`answer(final_answer: str)`** - Submit the final research answer.
- **`evaluate(rubric: list[dict])`** - Evaluate submitted answer using a structured rubric with weighted requirements.

### Rubric-Based Evaluation

The `evaluate` tool uses The LLM Data Company's [rubric](https://github.com/The-LLM-Data-Company/rubric/) package to grade answers against structured criteria with autograders.

## Setup

### Environment Variables

The environment requires several API keys and configuration:

**Required:**
- `EDGAR_IDENTITY` - Your identity for SEC EDGAR access (required by SEC regulations)
  - Format: `"Your Name your.email@example.com"`

**Optional:**
- `HUD_API_KEY` - For HUD telemetry and tracing
- `ANTHROPIC_API_KEY` - For Claude agent (if using Claude)
- `OPENAI_API_KEY` - For rubric evaluation (if using OpenAI-based autograders)

Set these before running `hud eval`:
```bash
export EDGAR_IDENTITY="Your Name your.email@example.com"
export ANTHROPIC_API_KEY="your-anthropic-key" # only if using an Anthropic model
export OPENAI_API_KEY="your-openai-key"
# Optional
export HUD_API_KEY="your-hud-key"
```

## Development

```bash
# Terminal 1 - Environment backend
cd environment
export SEC_EDGAR_USER_AGENT="your.name@example.com"
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
hud dev
```

## Tasks & Evaluation

```bash
# Build first in the global folder with the Dockerfile (creates rubrics:latest)
hud build
```

Your `tasks.json` uses `docker run` to launch the environment:

```json
{
  "prompt": "Analyze Tesla's FY2024 10-K filing...",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "rubrics:latest"]
    }
  },
  "evaluate_tool": {
    "name": "evaluate",
    "arguments": {
      "rubric": [...]
    }
  }
}
```

**Note:** Export environment variables before running. The Docker container will inherit them from your shell.

**Commands:**
```bash
# Build first
hud build

# Test task locally
export EDGAR_IDENTITY="Your Name your.email@example.com"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
hud eval tasks.json
# add the following flag to increase number of steps the agent can take (default is 10)
--max-steps 25

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
# hud.so/leaderboards/your-org/your-dataset
```

**Note**: Only public HuggingFace datasets appear as leaderboards!

📚 Learn more: [Creating Benchmarks](https://docs.hud.so/evaluate-agents/create-benchmarks) | [Leaderboards](https://docs.hud.so/evaluate-agents/leaderboards)

## Example Research Workflow

```python
# Agent searches for a company
company_info = search_company("TSLA")
# Returns: [{"ticker": "TSLA", "name": "Tesla Inc", "cik": "1318605"}]

# Agent gets recent filings
filings = get_filings("TSLA", form_type="10-K", limit=1)
# Returns: [{"filing_date": "2024-01-01", "form_type": "10-K", "description": "...", "filing_url": "..."}]

# Agent fetches filing content
content = get_filing_content(filings[0]["filing_url"])
# Returns: Full text content of the filing

# Agent submits final answer
answer("Based on Tesla's FY2024 10-K, revenue was $96.8B...")

# Evaluate answer using rubric
result = evaluate(rubric=[
    {"requirement": "Correctly states FY2024 revenue", "weight": 15},
    {"requirement": "Provides segment breakdown", "weight": 5},
])
# Returns: {"reward": float, "info": {"report": [...]}, "done": True}
```

## Dependencies

- **edgartools**: Python library for accessing SEC EDGAR data
- **fastapi**: Web framework for the environment server
- **httpx**: HTTP client for API calls
- **rubric**: LLM Data Company's rubric evaluation package
