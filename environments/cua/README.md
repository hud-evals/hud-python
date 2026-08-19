# Computer-Use Environment (HUD v6)

A HUD **v6** environment for **computer-use agents**: a virtual Linux desktop published as an **`rfb` (VNC) capability**.
The agent brings its own native computer-use tool and drives the screen; tasks grade the result
server-side with deterministic shell checks and an optional LLM judge.

## Layout

```
env.py           Environment: publishes the rfb capability and defines the cua_task template
tasks.py         task definitions (prompt + graders + slug)
supervisord.conf runs Xvfb, XFCE, Chromium, x11vnc, and the HUD server in one container
Dockerfile.hud   installs the desktop and environment dependencies
```

Supervisor runs the desktop as the unprivileged `ubuntu` user, so the on-screen terminal cannot
read the grading code. The HUD control server remains root.

## Run

Needs [uv](https://docs.astral.sh/uv/), Docker, and the HUD CLI.

```bash
uv sync
cp .env.example .env          # HUD_API_KEY
```

Deploy the image so each rollout receives a fresh desktop:

```bash
hud deploy
hud eval tasks.py claude --model claude-opus-4-7 --runtime hud --max-steps 100 -y
```

The multi-step task wants headroom (`--max-steps 100`). `hud eval` runs the first task only; add
`--full` for all three or `--task-ids <slug>`.

For a single local smoke rollout, build and attach to one container, then remove it:

```bash
docker build -f Dockerfile.hud -t hud-cua:dev .
docker run -d --name hud-cua --env-file .env -p 8765:8765 hud-cua:dev
hud eval tasks.py claude --task-ids open-website-example \
  --model claude-opus-4-7 --runtime tcp://127.0.0.1:8765 -y
docker rm -f hud-cua
```

One desktop cannot isolate concurrent or sequential rollouts, so the environment accepts one task
per container. Start a fresh runtime for every task instead of sharing a `tcp://` substrate.

## Tasks & grading

`env.py` defines one template, `cua_task`, instantiated per task in `tasks.py`:

```python
from env import cua_task

_my_task = cua_task(
    prompt="Navigate to example.com and report the page title.",
    bash_checks=[{"name": "browser_running", "command": "pgrep -f chromium", "weight": 0.3}],
    grading_criteria=["The agent correctly reports the page title"],
)
_my_task.slug = "my-task-slug"
```

| Knob | Type | How it scores |
|------|------|---------------|
| `bash_checks` | `list[{name, command, weight}]` | shell command run in the container (the desktop the agent drove), scored by exit code |
| `grading_criteria` | `list[str]` | rubric strings judged by an LLM (needs `HUD_API_KEY`) |

Each task's score is a weighted average of its checks, so it always stays in `[0, 1]`. When a task has
both bash checks and a judge, the bash checks together are worth half the score and the judge the other
half — so the agent can't pass on the judge alone without doing the work. With only bash checks, those
are the whole score.

Give each task variable a leading underscore (`_my_task`) and add it to the `tasks` list. Without the
underscore, `hud` finds the task twice — once as a top-level variable and once in the list — and fails
with a "duplicate slug" error.

> Adding a task needs **no redeploy** — it reuses the baked `cua_task` template, so the new prompt
> and graders travel at eval time. Redeploy only when `env.py`, the `Dockerfile`, or the desktop
> changes.

| Slug | Grading | What it tests |
|------|---------|---------------|
| `open-website-example` | bash + LLM | browser navigation, tagline identification |
| `create-document-example` | bash only | terminal use, deterministic file content |
| `shannon-multistep-research` | bash + LLM | long multi-hop research across pages, then a terminal write |

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
```
