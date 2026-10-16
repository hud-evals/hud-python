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

Deploy from this directory with:

```bash
hud deploy
```
