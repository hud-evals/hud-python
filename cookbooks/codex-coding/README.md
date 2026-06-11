# Codex Coding Agent

Build your own [Codex](https://github.com/openai/codex) with the HUD SDK: an
environment exposes an `ssh` capability backed by a `Workspace`, and
`OpenAIAgent` drives it with OpenAI's native `shell` and `apply_patch` tools —
the same protocol the `codex` CLI uses.

## Run

From this directory (requires `HUD_API_KEY` for gateway inference):

```bash
uv run codex_agent.py

# Custom task
uv run codex_agent.py --task "Create a Python script that prints the Fibonacci sequence"

# Custom working directory
uv run codex_agent.py --work-dir ./codex_output
```

To run the same environment as a packaged, sandboxed box instead of on your
machine, see `hud deploy` and `RemoteSandbox` in the deploy docs.
