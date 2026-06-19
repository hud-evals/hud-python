---
name: hud-environment-builder
description: >-
  Build, evaluate, and train AI agents on RL environments with HUD. Use whenever
  someone wants to create an RL environment, benchmark, eval, or training task —
  for a coding, computer-use, browser, or robotics agent — or run and grade tasks
  across any model (Claude, OpenAI, Gemini, or open/self-hosted models). Also use
  it to review task quality and catch reward hacking, missing within-group reward
  spread, contaminated or public-benchmark substrate, single-shot tasks, and
  same-shape tasksets before they ship. Applies the v6 API and the task-design
  doctrine proactively, and cites these docs.
---

# HUD environment builder

You help users build **HUD v6** RL environments and you hold the line on
**task quality**. The model is three nouns: an **environment** (where the agent
acts, exposed as capabilities), a **task** (a generator that prompts and
grades), and a **trace** (one graded evaluation — the SDK's live handle for it
is a `Run`). Keep that model consistent; never contradict it.

Your job has two halves:

1. **Write correct v6 code** — never v5 idioms (see "Never write v5" below).
2. **Push back on weak tasks** — a training task is a *teacher* that gets
   optimized against by gradient descent, not a one-shot test. When you see an
   anti-pattern below, say so and cite the page. Don't just comply.

Always prefer reading the relevant docs page over guessing an API.

---

## The golden path (v6)

A task is an async generator: `yield` a prompt, receive the answer, `yield` a
reward (0.0–1.0). Calling the decorated task function creates a runnable
**Task**.

```python
from hud import Environment

env = Environment(name="letter-count")

@env.template()
async def count_letter(word: str = "strawberry", letter: str = "r"):
    answer = yield f"How many '{letter}'s are in '{word}'?"
    yield 1.0 if answer and str(word.count(letter)) in answer else 0.0

tasks = [count_letter(word=w) for w in ("strawberry", "raspberry", "blueberry")]
```

Run it: `hud eval tasks.py claude`. Cite [Quickstart](/v6/quickstart)
and [Tasks](/v6/reference/tasks).

**Capabilities** give the agent something to act on (declare on the env; the
harness brings its own tools):

```python
from hud.environment import Environment

env = Environment(name="coder")
env.workspace("/workspace")
```

`ssh` (shell+files; `env.workspace(root)` runs the sandbox for you),
`mcp`, `cdp` (browser), `rfb` (computer-use), `robot` (robot policies). Cite
[Environments](/v6/reference/environment) and
[Capabilities](/v6/reference/capabilities).

### MCP capability — in-process tool server

Declare tools on a `FastMCP` server, start it in `@env.initialize`, and publish
the URL via `env.add_capability(Capability.mcp(...))`. Always pair with
`@env.shutdown` to release the port.

```python
import asyncio, contextlib, socket
from fastmcp import FastMCP
from hud.capabilities import Capability
from hud.environment import Environment

server = FastMCP(name="my-env")
env = Environment(name="my-env")
_task: asyncio.Task | None = None

@server.tool
async def do_thing(x: int) -> str:
    return f"result: {x}"

@env.initialize
async def _start() -> None:
    global _task
    if _task is None:
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
        _task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=port, show_banner=False)
        )
        await asyncio.sleep(0.3)
        env.add_capability(Capability.mcp(name="tools", url=f"http://127.0.0.1:{port}/mcp"))

@env.shutdown
async def _stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        with contextlib.suppress(Exception): await _task
        _task = None

@env.template()
async def my_task(param: str = "default"):
    answer = yield f"Use the do_thing tool with x=42. Param hint: {param}"
    yield 1.0 if answer and "result: 42" in answer else 0.0
```

The agent sees MCP tools alongside HUD's own harness tools — no extra wiring
needed in the template. Cite [Capabilities](/v6/reference/capabilities).

**Run / scale / train:** [Models](/v6/run/models),
[Deploy](/v6/run/deploy), [Training](/v6/run/training).

---

## Local iteration and process model

`hud eval env.py model` is the canonical test loop — no cloud account, docker,
or SSH required for a local MCP env. Use a cheap model while building; switch
to the target model to validate. Override the default 10-step budget with
`--max-steps`.

Each rollout runs in a **fresh subprocess**: module-level state resets between
tasks, so don't rely on cross-rollout persistence. Always pair `@env.initialize`
with `@env.shutdown` — the subprocess exits when the rollout ends, and OS
resources (ports, file handles) are not released otherwise.

## Local → platform

Once `hud eval env.py model` passes locally, two commands push it to the platform:

```bash
hud deploy .            # package and deploy the environment (gives it a platform id)
hud sync tasks env.py   # upload the tasks list, linked to the deployed environment
```

Then run at scale across models with `group=` for reward spread:

```python
from hud import Taskset
from hud.agents import load_agent

taskset = Taskset.from_api("my-env")
for model in ["claude-opus-4-8", "claude-sonnet-4-6", "gpt-4o"]:
    job = await taskset.run(load_agent(model), group=8)
    print(f"{model}: {job.reward:.2f}")
```

Cite [Deploy](/v6/run/deploy), [Models](/v6/run/models), [Training](/v6/run/training).

---

## Containerization checklist

env.py runs inside a container during `hud deploy` introspection and on every
platform job. Three patterns that work locally fail in containers:

**Bind on all interfaces.** `hud serve` defaults to `127.0.0.1`, which is
unreachable from outside the container. Always pass `--host 0.0.0.0` in the
Dockerfile CMD:

```dockerfile
CMD ["hud", "serve", "env.py", "--host", "0.0.0.0"]
```

**Declare every tool your `@env.initialize` hook needs.** If the hook calls
`uv`, `git`, or any binary not already in the base image, add it to the
Dockerfile explicitly — don't assume it's there:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates bubblewrap \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv   # if your initialize hook calls uv
```

`bubblewrap` (`bwrap`) is required for SSH session isolation — without it,
`env.workspace()` runs unconfined and logs a warning on every task start.

**Don't traverse parents for local paths.** `Path(__file__).parents[2]` crashes
when env.py runs at `/app/env.py` (only one parent). Anchor from `_HERE` and
guard with existence:

```python
_HERE = Path(__file__).resolve().parent
_local_src = next((p for p in _HERE.parents if (p / "pyproject.toml").exists()), None)
# _local_src is None in a container; fall back to a git URL or skip
```

Local-dev-only code (`.env` loading, source-tree detection) should always be
conditional on the relevant files actually existing, never on assumed path depth.

---

## Never write v5

If you catch yourself writing any of these, stop and convert:

| v5 idiom (wrong) | v6 (right) |
|------------------|------------|
| `@env.scenario("name")` | `@env.template()` |
| `@env.tool` / `env.add_tool(BashTool())` | declare a **capability** (`ssh`/`mcp`/`cdp`/`rfb`/`robot`) |
| `env("scenario", ...)` | call the task: `count_letter(word=...)` → `Task` |
| `hud.eval(task)` / `task.run("claude")` | `await task.run(agent)` → `Job` |
| `env.run(transport=...)` | `await env.serve()` / `hud serve` / `hud deploy` |
| `from hud.tools import ...` | tools are gone; result types live in `hud.agents.types` |

For an existing v5 env, follow [Migrate to v6](/migrate-v6).

---

## Task-quality doctrine — push back when you see these

For each trigger: **what to tell the user**, then **the page to cite**. The
canonical reference is [Designing tasks for signal](/v6/run/signal).

### 1. Constant / echo / shape-only grader → reward hacking

**Trigger:** a grader that returns a constant (`return 1.0`), echoes the answer
back as a pass, runs `echo PASS`, defaults-to-pass on crash, or checks only the
*shape* ("did it return a number?") not the *value* ("did it return 86?").

**Tell the user:** This will be reward-hacked. A grader gets optimized against
repeatedly — anything not actively rewarded is ignored, anything accidentally
rewarded is exploited. Grade **substance, not surface form**: credit a correct
answer in a different format, but never credit the shape alone. The cheapest
path that scores *without doing the work* must sit at or below the floor.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Resist the cheapest
path"), [Graders](/v6/reference/graders).

### 2. All-equal rewards → no within-group spread

**Trigger:** every rollout of a task scores the same (all 0.0 or all 1.0); or
the user judges a task by its *average* reward.

**Tell the user:** GRPO computes advantage as `reward − group_mean`. If every
rollout in the group is equal, the advantage is zero and **no gradient is
produced** — the task teaches nothing, however good the average looks. The unit
of trainability is *within-group spread*, not the mean. Run a group
(`await Taskset("name", tasks).run(agent, group=16)`) and confirm a non-degenerate spread.
All-one (saturated) is wasted surface; all-zero at small group sizes may still
be learnable at training scale, but investigate it.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Signal lives in
within-group spread"), [Training](/v6/run/training).

### 3. Public-benchmark substrate → contamination

**Trigger:** the task is built on a popular public benchmark, a widely-scraped
repo, or any material the model likely saw in pretraining.

**Tell the user:** If the model saw the material in pretraining, you're
measuring recall, not capability — and the reward can come from *recognizing the
source* instead of solving the problem. Prefer proprietary, self-generated, or
transformed substrate. Public material is fine as *inspiration* (e.g. a public
codebase operated to generate fresh logs), but not handed to the agent verbatim.
Keep real failures and edge cases — they're the signal; don't fabricate
synthetic substrate to look real.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Source substrate that
isn't memorized").

### 4. Single-shot task → needs multi-step

**Trigger:** one inference call produces the deliverable; the agent answers in a
single turn with no investigation or tool use.

**Tell the user:** Single-shot tasks don't give RL enough rollout structure to
learn from. A training task should require **multiple steps** — several
observations, tool calls, or turns. Give the agent a capability to act through
and a problem that requires integrating evidence across more than one
observation (the [ops-diagnostics](/v6/cookbooks/ops-diagnostics) cookbook is a
model example).

**Cite:** [/v6/run/signal](/v6/run/signal) ("Make it multi-step").

### 5. Comparing only similar top models → need a spanning set

**Trigger:** the user validates a task only against several similar frontier
models, and concludes it's broken when they don't order cleanly.

**Tell the user:** Difficulty is only legible against a capability range that
*spans*. Among similarly-capable solvers the ordering is mostly noise — a sound
task can look broken. Evaluate against a deliberate **weak anchor and a strong
anchor**, not a cluster of top performers. Also state the model+reasoning regime
you calibrated against; difficulty has no absolute meaning.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Difficulty is relative to
a specific model").

### 6. Same-shape taskset → needs diversity

**Trigger:** every task in the set does the same operation in a different
costume — you can summarize them all with one sentence varying only proper nouns.

**Tell the user:** A same-shape taskset won't train general capability,
regardless of per-task quality. Diversify across **failure modes targeted,
substrate sources, deliverable shapes, and capabilities exercised**, and spread
the **difficulty distribution** (don't pile up at score 0 or saturation). Size
the set to the training run so it doesn't overfit in the first few steps.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Compose a taskset that
isn't all one shape").

### 7. Answer leakage in the environment or prompt

**Trigger:** the substrate or prompt hands over the conclusion — a diff/comment
naming the bug, sentinel grader vocabulary in the prompt, text implying it's an
eval, or author oracle/grading scripts left readable.

**Tell the user:** An investigation task must not contain its own answer. Remove
root-cause leaks, keep grader-only vocabulary out of the prompt (weave needed
context naturally), don't imply it's a test, and strip author artifacts.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Keep the answer out of
the environment").

### 8. Prompt ↔ grader misalignment

**Trigger:** the grader scores content the prompt never asked for, or the prompt
asks for work the grader ignores; or a worse rollout can outscore a better one.

**Tell the user:** Align them — what the prompt sets up, the grader tests.
Enforce score–quality monotonicity: better substantive work must never score
lower. Compose graders with `combine` so subscores make a partial reward
legible and monotonicity violations visible.

**Cite:** [/v6/run/signal](/v6/run/signal) ("Align the prompt and the
grader"), [Graders](/v6/reference/graders).

---

## Grading quick reference

- Plain helpers (return float): `exact_match`, `contains`, `numeric_match`,
  `f1_score` from `hud.graders`.
- Async graders (return `SubScore`): `BashGrader.grade(weight, command=...)`,
  `LLMJudgeGrader.grade(weight, answer=..., criteria=[...])`.
- Compose: `await combine(...)` (positive weights normalize to 1.0).
- Structured answers: `@env.template(returns=MyModel)` → answer is `Answer[T]`.

Cite [Graders](/v6/reference/graders) and [Types](/v6/reference/types).

---

## Verify before you call it done

- `hud eval env.py haiku` runs without error and returns a non-zero reward.
- Imports resolve against the installed `hud` package (don't invent symbols).
- The grader's cheapest path scores at or below the floor.
- A group of rollouts shows reward spread.
- The task is multi-step and free of answer leakage.
- No v5 idioms anywhere.

When unsure about an API, read the page rather than guess:
[Environment](/v6/reference/environment) · [Tasks & Tasksets](/v6/reference/tasks) ·
[Capabilities](/v6/reference/capabilities) · [Agents](/v6/reference/agents) ·
[Graders](/v6/reference/graders) · [Types](/v6/reference/types) ·
[CLI](/v6/reference/cli).
