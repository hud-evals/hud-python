# Computer-Use Environment (HUD v6)

A HUD **v6** environment for **computer-use agents**: a virtual Linux desktop published as an **`rfb` (VNC) capability**.
The agent brings its own native computer-use tool and drives the screen; tasks grade the result
server-side with deterministic shell checks and an optional LLM judge.

## Layout

```
env.py           Environment: @env.initialize launches the desktop (as `ubuntu`) as subprocesses
                 + publishes the rfb capability; the cua_task grading template lives here too
tasks.py         task definitions (prompt + graders + slug)
Dockerfile.hud   desktop image (Xvfb · x11vnc · xfce4 · chromium) + v6 control channel (hud serve)
```

The desktop is launched by `env.py` (one code path for local `hud eval` and the packaged image) —
there is no init system. In the image it drops to the unprivileged `ubuntu` user so the agent's
on-screen terminal (uid 1000) can't read the `chmod 700` grading code; the control channel stays
root and can.

## Run

Needs [uv](https://docs.astral.sh/uv/), Python 3.11/3.12, and the HUD CLI. The virtual desktop is
`Xvfb` + `x11vnc` — **X11, so Linux-only**. macOS is Quartz/Cocoa with no local X server, so use
Docker there.

```bash
uv sync
cp .env.example .env          # HUD_API_KEY
```

**Local, dockerless — Linux, fastest iteration.** `@env.initialize` spawns the desktop itself, so
`hud eval` runs the whole env as a local child process, no Docker, no build:

```bash
sudo apt install -y xvfb x11vnc chromium xfce4   # the desktop the env spawns
hud eval tasks.py claude --task-ids open-website-example -y --max-steps 100
```

**Docker — required on macOS, and the packaging/deploy path.** Build once, then run a container
that serves the env + the in-container judge and attach the agent over `tcp://` (below):

```bash
docker build -f Dockerfile.hud -t cua-template:dev .
docker run -d --env-file .env -p 8765:8765 cua-template:dev
```

Point a computer-use agent at the served env. The multi-step task wants headroom (`--max-steps 100`);
`hud eval` runs the first task only — add `--full` for all three or `--task-ids <slug>`:

```bash
hud eval tasks.py claude --model claude-opus-4-7 --runtime tcp://127.0.0.1:8765 --max-steps 100 -y
```

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
uv run pytest tests/ -q
```