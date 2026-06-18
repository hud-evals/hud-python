# LocalRuntime orphan-process bug (FIXED)

Status: fixed in `hud/eval/runtime.py`. Repro: `hud/eval/tests/test_local_runtime_orphan.py`.

## What was wrong (two missing lines)

`LocalRuntime` spawns the server child with:

```python
proc = await asyncio.create_subprocess_exec(*cmd, stdout=..., cwd=...)
```

No `start_new_session=True` → the child inherits the **parent's process group**.

On teardown `_terminate` does:

```python
proc.terminate()   # os.kill(child_pid, SIGTERM) — one pid, period
```

`os.kill(pid, ...)` signals exactly that pid.  
Any subprocess the env's `@env.initialize` hook spawns is a *grandchild* living in the same inherited process group but **not reachable by a single-pid signal**. The direct child dies; the grandchild keeps running, re-parented to init — orphaned.

## The fix (two changes)

**1. Spawn the child in its own session** — `start_new_session=True` runs
`setsid()`, giving the child a fresh process group (pgid == its pid) that all
its descendants inherit. This also detaches it from the terminal.

```python
proc = await asyncio.create_subprocess_exec(
    *cmd, stdout=asyncio.subprocess.PIPE, cwd=..., start_new_session=True,
)
```

**2. Signal the whole group on teardown** — not just the root pid. Because the
child is the group leader, `proc.pid` *is* the pgid, so no `getpgid` lookup is
needed (and we avoid racing a just-exited child):

```python
with contextlib.suppress(ProcessLookupError):
    os.killpg(proc.pid, signal.SIGTERM)          # graceful: child + grandchildren
try:
    await asyncio.wait_for(proc.wait(), 10.0)
except TimeoutError:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)      # escalate stragglers
    await proc.wait()
```

`os.killpg` signals every process whose pgid matches — the direct child, its
grandchildren, and any further descendants — so nothing survives teardown.
(Windows has no `killpg`; there it falls back to `proc.terminate()/kill()`.)

## Ctrl+C

`start_new_session=True` takes the child out of the terminal's foreground group,
so a Ctrl+C delivers SIGINT to the orchestrator **only**. That raises
`KeyboardInterrupt`, which unwinds through the `async with` and runs
`_terminate` in the `finally` — the same group-SIGTERM-then-SIGKILL path. So
Ctrl+C tears the whole tree down gracefully instead of leaving the child to
catch a stray SIGINT on its own.

## Why it wasn't caught earlier

Envs that only do in-process work (pure Python, no `subprocess.Popen` / `asyncio.create_subprocess_exec` inside `@env.initialize`) don't spawn grandchildren, so the bug is invisible. It surfaces only when an env boots a real OS process as part of its lifecycle — simulators, MCP servers, robot stacks, etc.
