# Terminal-Bench on HUD

Run [Harbor](https://github.com/harbor-framework/harbor)-format task directories —
Terminal-Bench 2.0 and anything else in that layout — through the HUD runtime.
The experimental `hud.integrations.harbor` adapter builds each task's own
environment image, wraps it with the HUD control channel, and returns a
`Taskset` you run like any other.

## Run

```bash
uv run run.py ./terminal-bench
```

Run commands from this cookbook directory so uv uses its local project.

By default, the runner uses each task's `solution/solve.sh` as an oracle. Run
the oracle first to verify that the environment and verifier can produce the
task's expected reward before evaluating a model.

```bash
# A real agent, once the oracle is clean
uv run run.py ./terminal-bench --agent claude-sonnet-4-5

# Just the ones you are debugging
uv run run.py ./terminal-bench --task qemu-startup --task configure-git-webserver
```

Output is one row per task plus a mean over the tasks that actually finished:

```
task                                      reward  status
configure-git-webserver                      1.0  completed
qemu-startup                                 1.0  completed
regex-log                                    1.0  completed

scored 3/3 (errored 0), mean reward 1.00
```

Runs that end in `error` are reported separately from scored runs.

## What you need

- Docker. `DockerRuntime` gives every container the profile the in-image
  workspace sandbox needs, so no extra flags.
- Network, for the image builds and for whatever the tasks fetch.
- `HUD_API_KEY` only when `--agent` names a model, for gateway inference.

The first run builds two images per distinct environment (the task's own, then
the HUD wrapper) and is slow; later runs reuse the layers.

## Use a local SDK build

The adapted image installs `hud` from PyPI by default. To use a local SDK
build, build a wheel and pass it to the adapter:

```bash
uv build --wheel --out-dir /tmp/hud-wheel ../..
uv run run.py ./terminal-bench --hud-requirement /tmp/hud-wheel/hud-*.whl
```
