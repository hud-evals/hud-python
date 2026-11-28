# Blank Environment

A minimal HUD environment with an exact-match task and no capabilities.

```bash
uv sync
hud task list
```

No built-in agent runs this environment yet: the built-in agents run a coding CLI in an `ssh`
workspace, and this environment has none. Run it with
[your own harness](https://docs.hud.ai/v6/advanced/extending#bring-your-own-harness).

`env.py` defines the task template and grader. `tasks.py` binds two concrete task rows.
