# Browser Use

A visual web-research environment backed by a real Chromium browser. Agents can
inspect screenshots, navigate, click, scroll, and use the browser's developer
protocol without leaving the allowed website.

The featured task asks the agent to inspect rental photos and count the chairs
at a specific kitchen table. Grading checks both the exact answer and whether
the agent cited screenshot evidence that actually showed the kitchen.

The environment exposes:

- `cdp`: direct browser control for compatible agents
- `mcp`: screenshot and browser-interaction tools

## Run the example task

```bash
uv venv
uv pip sync requirements.txt
uv run --no-project hud eval tasks.py claude --runtime local -y
```

`tasks.json` is the task source for evaluation and taskset sync; `tasks.py` binds it to the local environment. To adapt it,
add an `@env.template` in `env.py`, define the task goal and allowed website, and
check the answer against the screenshots collected during the task. Keep the
browser tools and evidence checks shared between tasks.

## Deploy

Deploy from this directory with:

```bash
hud deploy
```
