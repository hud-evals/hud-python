# Coding Environment (HUD v6)

A coding environment: the agent gets a git repo over a sandboxed **`ssh` workspace** (the
harness brings its own bash/file tools), and grading is diff-based — capture the agent's
changes, reset to the pre-agent snapshot, re-apply the diff, bring in the hidden tests, run
them. Generic tasks, SDLC workflow tasks, and SWE-bench Pro are flavors on this core.

The agent never sees the answer key. Task setup moves the repo's real `.git` — which may hold
solution branches or the fix commit — into a vault outside the workspace and leaves a fresh
single-commit repo: git works normally, but there is no history, no refs, and no remotes.
Grading discards the agent's `.git`, restores the vault, and checks hidden tests out *after*
the agent's diff. In images, the vault and instance assets live under `/hud` (root, mode 700)
and agent shells drop to a non-root uid via `setpriv`.

This directory is a self-contained uv project — run every command below from it (`hud init`
hands you a copy of it as your own environment package).

## Layout

- `env.py` — the environment: workspace wiring plus the three task templates.
- `coding/repo.py` — the shared repo lifecycle: vault, snapshot, diff capture, reset, apply.
- `coding/github.py` — mock GitHub for SDLC tasks: issue/PR store served as `github_*` tools.
- `coding/swe_bench_pro.py` — the SWE-bench Pro grading pipeline.
- `tasks.py` — sample generic and SDLC tasks.
- `swe_tasks.py` — the SWE-bench Pro task source: fetches instances and builds their images
  when run; task rows when imported.
- `Dockerfile.hud` — the image definition for both flavors: `BASE` selects the generic
  repo-clone head or a prebuilt instance image.

## Generic tasks (`coding-task`)

Point the env at a repo (`REPO_URL`; locally it clones per process, or bake it with
`Dockerfile.hud`) and parameterize the template: a `base_ref` to start from, a `test_ref` whose
`test_files` are the hidden tests (checked out from the vaulted history at grade time), and a
`test_command` scored by exit code. Tasks follow the 3-branch convention — `{task}_baseline` /
`{task}_test` / `{task}_golden`; `tasks.py` ships four sample bugs on
[coding-template-sample](https://github.com/hud-evals/coding-template-sample):

```bash
uv sync
hud set HUD_API_KEY=your-key-here
hud eval tasks.py claude --task-ids sentry-fix -y --runtime local
```

## SDLC tasks (`sdlc-task`)

The generic flavor plus workflow: the repo gets an `origin` remote (a bare mock-GitHub repo the
agent pushes to) and `github_*` MCP tools seeded with the task's issues. The deliverable is a
pushed branch with a pull request — grading checks the PR head out of the remote, brings in the
hidden tests, runs the test command (weight 0.8), and scores the PR itself (0.2): a structural
title/body check by default, or `pr_rubric` judged by `LLMJudgeGrader` when provided. See the
`sentry-fix-pr` sample in `tasks.py`:

```bash
hud eval tasks.py claude --task-ids sentry-fix-pr -y --runtime local
```

## SWE-bench Pro tasks

Each of the 731 public [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) instances
ships a prebuilt image (`jefzda/sweap-images:<tag>`, `linux/amd64`) with the repo and toolchain
baked in. Running `swe_tasks.py` fetches the dataset row plus the official
`run_script.sh`/`parser.py` into `instances/<id>/` and builds `Dockerfile.hud` with the
instance's image as `BASE`, so the image serves this env from inside. Grading replays the
official evaluator: resolved iff every `fail_to_pass` **and** `pass_to_pass` test passes.

```bash
uv run swe_tasks.py instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan
hud eval swe_tasks.py claude --task-ids nodebb-04998908
uv run swe_tasks.py <id>... --push registry.io/acme   # push for cloud runtimes
```

## Tests

```bash
uv run pytest tests/ -q --ignore=tests/test_integration.py   # offline + hermetic local e2e
uv run pytest tests/test_integration.py -v                   # SWE-bench gold-patch check (Docker)
```

`test_local_rollout.py` runs the generic and SDLC flavors end to end against a fixture 3-branch
repo (no Docker or network): the golden ref grades 1.0 and the untouched baseline 0.0. The
integration suite is the same check for built SWE-bench Pro instances.

## Caveats

- Public benchmarks are public: a networked agent could fetch solutions from GitHub. Disable
  network egress at the runtime layer if that matters for your run.
- Non-root local runs require bubblewrap. If it is unavailable, serving fails rather than
  exposing vaulted answer-key refs to the agent process.
- The uid wall needs `setpriv` (util-linux) in the image; the repo path comes from `REPO_DIR`
  (`/app` in instance images).
- The agent runs as uid 1000, so the baked repo must belong to it. The workspace only chowns
  its own directory at start (O(1), keeps boot fast); the tree is owned where it's staged —
  the generic build chowns `/app` in the clone step, and task setup re-chowns after root
  mutates the worktree (checkout, vaulting).

## Documentation

See the [full docs](https://docs.hud.ai) for tasks, evaluation, and scaling.
