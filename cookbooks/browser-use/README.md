# browser-use

Run [browser-use](https://github.com/browser-use/browser-use) as a HUD agent. It
attaches to the browser an environment publishes as a `cdp` capability and runs its
own loop; the run records browser-use's final answer.

`agent.py` is the whole adapter: copy it as a template for wrapping another agent
framework over the capability it needs (`ssh`, `mcp`, `rfb`, `robot`).

## Run

From this directory, with `ANTHROPIC_API_KEY` set:

```bash
uv run run.py path/to/tasks.py
```

The tasks' environment must publish a browser, e.g.
`Capability.cdp(url="ws://127.0.0.1:9222/devtools/browser/...")`.
