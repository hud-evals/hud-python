# RL environment on Daytona sandboxes

Run an agent evaluation where every rollout gets its own fresh
[Daytona](https://daytona.io) sandbox. One `env.py` runs locally and on
Daytona unchanged.

The task: a small Python file with seeded bugs, graded by whether it passes
its tests.

The full measured walkthrough — spin-up ladders to 256 concurrent, warm pools,
and sizing rules — is [the guide on Daytona's docs](https://www.daytona.io/docs/guides/reinforcement-learning/hud-rl-cookbook). The
condensed HUD-side walkthrough is
[on the HUD docs](https://docs.hud.ai/v6/cookbooks/daytona-rl).

| File | Purpose |
|------|---------|
| `env.py` | The environment: a workspace, one task, a grader |
| `bugs.py` | 24 deterministic bug variants (4/5/6 broken functions) |
| `tasks.py` | Task list for `hud eval` |
| `pool.py` | Warm-pool helpers (create, wait until really full, drop), used by `bench/warmpool.py` |
| `snapshot.py` | Content-addressed snapshot naming |
| `bench/` | Repro receipts behind the guide's numbers |

## Setup

```bash
uv sync
hud set HUD_API_KEY=...        # https://hud.ai/project/api-keys
export DAYTONA_API_KEY=...     # https://app.daytona.io
```

`DAYTONA_API_KEY` has to be a real `export`: the Daytona SDK reads the process
environment, not `~/.hud/.env`.

## Run

```bash
hud eval tasks.py claude                                   # one local rollout, no Daytona
PYTHONPATH=. uv run bench/baseline.py --max-concurrent 24  # every variant on Daytona
PYTHONPATH=. uv run bench/reap.py --delete                 # clean up stray sandboxes
```

`DaytonaRuntime` builds the snapshot from `Dockerfile.hud` on first run and
reuses it after.

The snapshot name is a hash of `env.py`, `bugs.py`, `Dockerfile.hud` and
`pyproject.toml` (`snapshot.py`), so editing the environment mints a new one
instead of silently running the old image.

## `bench/`

Run from this directory with `PYTHONPATH=.`.

| File | Purpose |
|------|---------|
| `baseline.py` | Pass rate across the bug variants, by bug count |
| `run_daytona.py` | One rollout on Daytona, with the spin-up/agent time split |
| `ladder.py`, `ladder_reps.csv` | Concurrency ladder, N up to 256 |
| `warmpool.py`, `warmpool_repro.py` | Warm-pool A/B |
| `reap.py` | Delete stray sandboxes and stale snapshots |

## Two things that will bite you

**A warm pool says it's full before it is.** Its `current_size` reports the target
about 12 seconds early. `pool.py` counts actual sandboxes instead — and that
count needs `include_warm=True`, because unclaimed pool members are excluded from
`list_sandboxes` by default. Count the obvious way and you get 0 forever.

**An interrupted run can strand a pool.** A hard kill of a script that created
a warm pool leaves the pool parked. `bench/reap.py` will not
catch it — it calls `AsyncDaytona.list()`, blind to unclaimed pool sandboxes for
the same `include_warm` reason, and it does not delete pools at all. Check after
any interrupted run.
