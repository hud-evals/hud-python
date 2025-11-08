# SEC EDGAR Rubrics Environment

SEC filing research environment powered by the SEC EDGAR database for accessing company filings and financial data, with rubric-based evaluation for structured grading provided by [The LLM Data Company](https://llmdata.com).

See [docs](https://docs.hud.ai/build-environments) for the complete environment design workflow.

## Architecture

**`environment/`** - Manages SEC EDGAR and web search integration
- Uses the edgartools Python library to access SEC filing data
- Integrates with Exa API for supplementary web search capabilities
- Exposes HTTP endpoints for research workflows with exponential backoff for rate limiting

**`server/`** - Wraps data in MCP tools
- Provides research tools for agents to access SEC filings, financial data, and web search
- Agents and tasks interact only with these tools

**Why separate?** Edit tools for the agent or tasks without restarting the environment backend.

## Tools

### SEC EDGAR Tools
- **`setup()`** - Initialize the environment and reset state.
- **`search_company(query: str)`** - Search for a company by ticker symbol or name. Returns company information including ticker, name, and CIK.
- **`get_filings(ticker?: str, form_type?: str, limit?: int, cutoff_date?: str)`** - Get SEC filings. When `ticker` is provided, returns company-specific filings. Otherwise, returns global recent filings. Can filter by form type (e.g., "10-K", "10-Q", "8-K"), limit results, and filter by date (YYYY-MM-DD).
- **`get_filing_content(filing_url: str)`** - Fetch the full text content of a specific SEC filing from its URL.
- **`get_financial_data(ticker: str, accession_number: str)`** - Extract financial statements and key metrics from a 10-K or 10-Q filing. Returns income statement, balance sheet, cash flow, and other financial data.
- **`get_segment_data(ticker: str, accession_number: str)`** - Extract segment-level financial data from a 10-K or 10-Q filing for companies with multiple business segments.
- **`get_filing_sections(ticker: str, accession_number: str)`** - Extract specific sections from a 10-K or 10-Q filing (e.g., Business, Risk Factors, MD&A).

### Web Search Tools
- **`web_search(query: str)`** - Search the web using Exa API. Returns titles and URLs of relevant results.
- **`web_fetch(url: str)`** - Fetch and extract content from a web URL. Returns summary, highlights, and full content.

### Evaluation Tools
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
- `EXA_API_KEY` - For web search and content fetching capabilities (if using web_search/web_fetch tools)
- `HUD_API_KEY` - For HUD telemetry and tracing
- `ANTHROPIC_API_KEY` - For Claude agent (if using Claude)
- `OPENAI_API_KEY` - For rubric evaluation (if using OpenAI-based autograders)

Add these to your .env before running `hud eval`:
```bash
export EDGAR_IDENTITY="Your Name your.email@example.com"
export EXA_API_KEY="your-exa-key" # optional, for web search
export ANTHROPIC_API_KEY="your-anthropic-key" # only if using an Anthropic model
export OPENAI_API_KEY="your-openai-key"
# Optional
export HUD_API_KEY="your-hud-key"
```

## Development

```bash
# Terminal 1 - Environment backend
cd environment
export EDGAR_IDENTITY="Your Name your.email@example.com"
export EXA_API_KEY="your-exa-key"  # optional, for web search
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
export EXA_API_KEY="your-exa-key"  # optional, for web search
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
hud eval tasks.json --max-steps 25

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
# Initialize environment
setup()

# Agent searches for a company
company_info = search_company("TSLA")
# Returns: [{"ticker": "TSLA", "name": "Tesla Inc", "cik": "1318605"}]

# Agent gets recent filings
filings = get_filings(ticker="TSLA", form_type="10-K", limit=1)
# Returns: [{"filing_date": "2024-01-01", "form_type": "10-K", "accession_number": "...", "filing_url": "..."}]

# Agent extracts financial data
financial_data = get_financial_data(ticker="TSLA", accession_number=filings[0]["accession_number"])
# Returns: {"has_financials": True, "financial_data": {...income statement, balance sheet, etc...}}

# Agent gets specific sections from the filing
sections = get_filing_sections(ticker="TSLA", accession_number=filings[0]["accession_number"])
# Returns: {"sections": {"business": "...", "risk_factors": "...", "mda": "..."}}

# Agent uses web search for additional context
search_results = web_search("Tesla FY2024 revenue analysis")
# Returns: [{"title": "...", "url": "..."}]

# Agent fetches web content
web_content = web_fetch(search_results[0]["url"])
# Returns: "=== SUMMARY ===\n...\n=== KEY HIGHLIGHTS ===\n...\n=== FULL CONTENT ===\n..."

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
- **Exa API**: Web search and content extraction (optional, for web_search/web_fetch tools)

## Acknowledgments

* [EdgarTools](https://github.com/dgunning/edgartools) - Python library to access SEC EDGAR
* [SEC EDGAR MCP](https://github.com/stefanoamorelli/sec-edgar-mcp) - Rich OSS SEC MCP server
