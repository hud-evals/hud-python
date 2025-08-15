<div align="left">
  <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 20px;"/>
</div>

<h3>
Evaluate your Computer Use AI agents across web browsers, desktop environments, and custom scenarios.
</h3>

### 🚀 Are you a startup building agents?

[📅 Hop on a call](https://cal.com/jay-ram-z6st6w/demo) or [📧 founders@hud.so](mailto:founders@hud.so)

We're here to help with eval strategies, custom environments, or improving your agent architecture!


> **Early Release Notice**: We'd love to hear your feedback in [Issues](https://github.com/hud-evals/hud-python/issues), as the SDK is still evolving!

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

## ✨ What You Can Do

**[Evaluate Existing Benchmarks](https://docs.hud.so/examples/benchmarking-agents)**
```python
from hud import load_taskset, run_job, ClaudeAgent

taskset = await load_taskset("WebVoyager")  # or GAIA, OSWorld-Ubuntu, Mind2Web
job = await run_job(ClaudeAgent, taskset, "my-evaluation")
```

**[Create Custom Tasks](https://docs.hud.so/task-creation)** 
```python
from hud.task import Task

task = Task(
    prompt="Find and book the cheapest flight from NYC to Paris",
    gym="hud-browser",
    setup=("goto", "https://kayak.com"),
    evaluate=("page_contains", "confirmation")
)
```

**[Build Custom Environments](https://docs.hud.so/environment-creation)**
```python
from hud.types import CustomGym

# Launch any website as an environment
custom_gym = CustomGym(
    image_or_build_context="nginx:alpine",
    location="local"
)

# Or create complex Docker environments - see environments/ folder for examples
```

**[Trace Tool Calls Alongside HUD Environments (or Independently)](https://docs.hud.so/examples/mcp-agent-tracing)**
```python
import hud

with hud.trace("my-agent-run"):
    # Your agent code here - MCP calls automatically captured
    result = await agent.run(task)
```

## Quick Start

### Installation

```bash
pip install hud-python
```

### API Key Setup

Before getting started, you'll need to obtain an API key:

1. Visit [app.hud.so](https://app.hud.so) to create a free account and generate your API key
2. Set it in your environment or .env file:

```bash
export HUD_API_KEY=your_api_key_here
```

### Simple Browser Example with Claude Computer Use

> This example uses the `@register_job("test-run")` decorator, so the results of this run will appear under the job named "test-run" on the your [HUD Jobs page](https://app.hud.so/jobs).

Make sure your have defined your `ANTRHOPIC_API_KEY` in environment variables to run Claude.

```python
import asyncio
from hud import gym, register_job
from hud.task import Task
from hud.agent import ClaudeAgent

@register_job("test-run")
async def main():
    task = Task(
        prompt="Insert the text 'capybara' into the search bar",
        gym="hud-browser",
        setup=("goto", "google.com"),
        evaluate=("contains_text", "capybara")
    )
    print(f"Running task with prompt: {task.prompt}")
    
    # Create environment using the gym module
    env = await gym.make(task)
    
    # Initialize Claude agent (API key is loaded automatically)
    agent = ClaudeAgent()
    
    # Agent loop with predict and step functions
    obs, _ = await env.reset() # Gets first observation
    for i in range(5):
        actions, done = await agent.predict(obs)
        print(f"Agent action {i}: {actions}")

        obs, reward, terminated, info = await env.step(actions)
        if done or terminated: break
    
    # Evaluate and close
    result = await env.evaluate()
    print(f"Evaluation result: {result}")
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Alternatively, run a full evaluation set via the ```run_job``` command:

```python
from hud import load_taskset, run_job, ClaudeAgent

# Load a benchmark
taskset = load_taskset("GAIA")

# Evaluate
job = await run_job(ClaudeAgent, taskset, "test-gaia-job")

# Get results OR view them in app.hud.so
print(await job.get_analytics())
```

## Ready-to-Use TaskSets

- **WebVoyager** - Web navigation and interaction
- **Mind2Web** - Complex web application tasks  
- **GAIA** - Question answering and reasoning
- **OSWorld-Ubuntu** - Desktop interaction
- **hud-samples** - Getting started examples

## Community

**Contributing Custom Environments**

Add your environment to the `environments/` folder and submit a PR! Examples:
- `environments/novnc_ubuntu/` - Ubuntu with VNC access 
- `environments/pokemon_controller/` - Pokemon emulator environment (In Development)
- `environments/qa_controller/` - Lightweight app sample

See [Custom Environments Guide](https://docs.hud.so/environment-creation) for details.

## Documentation Sections

Explore the core concepts and features of the SDK:

*   **[Task Creation](https://docs.hud.so/task-creation)**: Build custom evaluation scenarios with setup and evaluation criteria.
*   **[Environments](https://docs.hud.so/environments/browser)**: Understand browser environments and create custom Docker-based environments.
*   **[Agents](https://docs.hud.so/concepts/agent)**: Learn about the agent architecture (Claude, Operator) and how they process observations and predict actions.
*   **[Jobs](https://docs.hud.so/concepts/job)**: Group related runs for analysis and viewing on the HUD platform.
*   **[MCP Telemetry](https://docs.hud.so/telemetry/mcp)**: Automatic tracing of Model Context Protocol interactions.
*   **[Full API Reference](https://docs.hud.so/api-reference/gym)**: Detailed specifications for all modules and classes.

## [Examples](examples/)

We recommend you first take a look at the example notebooks showing how to use the HUD SDK:

1. [Browser Basics](examples/environments/simple_browser_example.py) - Simple browser interaction with live view
2. [Task Design](examples/evaluations/tasks.ipynb) - Creating and customizing tasks
3. [OSWorld](examples/evaluations/osworld.ipynb) - Running the OSWorld benchmark
4. [Local Development](examples/environments/gmail_local.ipynb) - Setting up local custom environments

## Documentation

For comprehensive guides, examples, and API reference, visit [our docs](https://docs.hud.so/introduction)

## License

[MIT License](LICENSE)

## Citation

If you use this SDK in your research, please cite it as follows:

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Oskars Putans and Govind Pimpale and Mayank Singamreddy and Nguyen Nhat Minh},
  title = {{HUD: An Evaluation Platform for Agents}},
  date = {2025-04},
  url = {https://github.com/hud-evals/hud-python},
  langid = {en}
}
```
