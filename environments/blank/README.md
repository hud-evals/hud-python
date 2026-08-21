# Blank Environment

A minimal HUD environment with an exact-match task and no capabilities.

```bash
uv sync
hud eval tasks.py claude --runtime local -y
```

`env.py` defines the task template and grader. `tasks.py` binds two concrete task rows.
