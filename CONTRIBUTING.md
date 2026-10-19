# Contributing to HUD

We welcome contributions to the HUD SDK! This guide covers how to get started.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/hud-python` (the repo keeps its historical name; the PyPI package is `hud`)
3. Install [uv](https://docs.astral.sh/uv/) and set up dev dependencies:
   ```bash
   cd hud-python
   uv sync --extra dev
   ```

## Development Workflow

### Running Tests

CI runs the full suite; locally, run what you touched:

```bash
uv run pytest tests/cli -n auto        # one area, in parallel
uv run pytest -n auto                  # everything not marked e2e
```

Tests run on Python 3.11 and 3.12 in CI. The template scenarios in
`tests/e2e/test_templates.py` take each `hud init` template to a graded rollout
and install its packages, so they need network access to PyPI.

Tests that need something outside the process are marked `e2e` and run only
when selected. They skip themselves when what they need is missing:

```bash
uv run pytest -m "e2e and docker"                  # a running Docker daemon
uv run pytest -m "e2e and sandbox"                 # Linux, root and bubblewrap
uv run pytest -m "e2e and live and not hosted"     # HUD_API_KEY; spends credits
uv run pytest -m "e2e and hosted"                  # deploys and runs on the HUD platform
uv run pytest -m "e2e and modal"                   # a Modal account
uv run --with "lerobot[dataset]" pytest -m "e2e and lerobot"  # real LeRobot datasets
uv run pytest -m e2e tests/e2e/test_cookbooks.py   # the cookbooks; CI checks them nightly
```

### Writing Tests

Tests are black-box: they drive the SDK through a public boundary and fake only
what lies outside it. `tests/support` holds the fakes, and `tests/conftest.py`
provides them as fixtures:

- `hud_env`: the SDK's configuration. Every test starts with a temporary home,
  no credentials, and service URLs pointed at a closed port; `hud_env.set(...)`
  changes environment variables the way a user would.
- `platform`: the HUD platform, signed in. Seed what a test needs
  (`platform.add_taskset(...)`, `platform.add_model(...)`) and check what the
  code left behind (`platform.tasksets`, `platform.builds`, `platform.traces`).
  `reporting` is the same platform with telemetry uploads on.
- `services`: the HTTP layer under `platform`, for services the platform fake
  does not model and for faults: `platform.services.route(...)` overrides a
  route; anything undeclared answers 404.
- `models`: scripted model providers behind the fake gateway, speaking OpenAI
  Chat Completions and Responses, Anthropic Messages and Gemini. Script turns
  with `models.script([call("bash", command="ls"), say("done")])`.
- `hud`: the real `hud` CLI in a subprocess. `hud("jobs", "list", "--json")`
  returns the exit code, stdout and stderr.
- `fake_docker`: a `docker` executable on `PATH` that logs every invocation and
  answers from rules, an image store (`images`), or container directories
  (`rootfs`).
- `ScriptedAgent`, `RecordingProvider`, `served`, `fake_screen`, and `steps` for
  agents with fixed answers, observed placements, served environments, a VNC
  screen, and the step spans a run exported.
- `fake_browser`, `control_peer`, and `relay` for a DevTools endpoint, a scripted
  control-channel peer, and a TCP relay that severs, refuses or stalls
  connections; `eventually` polls for a condition instead of sleeping.

For example:

```python
async def test_a_graded_rollout_reports_its_reward(reporting):
    job = await Taskset("demo", [row]).run(ScriptedAgent("3"), runtime=LocalRuntime(env))

    (trace,) = reporting.traces.values()
    assert (job.reward, trace.exit["reward"]) == (1.0, 1.0)
```

The suite is organized by contract. A contract is something the SDK owes a
user or a peer, such as "a rollout that runs past its deadline is cancelled and
leaves its placement clean" or "a failed command prints one JSON error document".
Each contract has one scenario, written as a table, in a file named for it:

- `tests/environment/`: the graders a task author uses, in graded rollouts.
- `tests/protocol/`: the control channel and the other wire protocols an
  environment serves, frame by frame: its server, its client, sessions, error
  replies, tunnels and file tracking.
- `tests/rollout/`: what every runtime owes a rollout (outcomes, deadlines and
  cancellation, refusals, the verifier, what the agent sees). The runtimes are
  the rows, listed in `tests/rollout/placements.py`; the other files hold the
  scheduler, task sources, taskset files, and what only one runtime does.
- `tests/agents/`: what every agent family owes its run (prompt and tools, the
  tool loop, limits, provider faults, credentials, a fixture taken to its
  reward), with the families listed in `tests/agents/families.py`, and one wire
  file per family for what only that family sends.
- `tests/cli/`: what every `hud` command owes its caller (`test_changes`,
  `test_output`, `test_platform_effects`, `test_shell`), with the command tree
  listed in `tests/cli/commands.py`, and one file per command for what only that
  command does.
- `tests/capabilities/`, `tests/reporting/`, `tests/workspace/`, `tests/robot/`,
  `tests/integrations/`, `tests/train/`: capability clients, what the platform
  and the trace record, the agent's workspace, the robot stack, external task
  formats, and training batches.
- `tests/` itself: what the whole SDK owes: importing it, its exported names,
  and where its configuration comes from.
- `tests/e2e/`: the `hud init` templates and the cookbooks.

When a contract has several implementations, they are rows in its table, drawn
from the product's own registry: the runtimes `hud.eval` exports, the Typer
command tree, `hud.capabilities.__all__`, the agent families. A completeness
test holds each table to its registry, so a new implementation fails the suite
until it has a row, or an entry in the table's not-applicable list with the
reason. The backing is a row's parameter too: the same scenario runs on the
in-process fake by default, and its rows on Docker, Modal or a real account
carry the lane's marks.

Rules, enforced by `scripts/check_tests.py` and in review:

- Never patch a `hud` name, import a private one, or touch a private attribute.
- Never hand-build what a producer emits; run the producer.
- Write one scenario per contract as a table, and add a row for a new case or a
  new implementation. Test the contract once, at the outermost boundary where it
  is observable.
- Keep a test for one implementation only for behavior no other implementation
  has, such as a provider's wire translation.
- Write no unit tests. A test drives a scenario (a command, a rollout, a served
  environment, a wire peer) and checks what it left behind, never what a
  function returns.
- Assert the fields a case is about, not a recorded copy of a whole output: no
  snapshots of argv, request bodies or rendered text.
- Do not commit a test to check your own change; use a temporary script.

### Code Quality

```bash
uv run ruff format . --check   # Formatting
uv run ruff check .            # Linting
uv run --extra dev --extra train ty check  # Type checking
uv run python scripts/check_tests.py       # Tests stay black-box
```

## Code Style

- Python 3.11+ features are allowed
- Type hints required for public APIs
- Line length limit: 100 characters
- Follow existing patterns in the codebase

## Pull Request Process

1. **Branch naming**: `feature/description` or `fix/issue-number`
2. **Commits**: Use clear, descriptive messages
3. **Tests**: All CI checks must pass (ruff, ty, the test policy check, pytest)
4. **Review**: Address feedback promptly

## Releases

1. Set the intended package version on `main`, then manually run the **Pre-release** workflow on `main`.
2. After it succeeds, tag that exact commit and publish the GitHub release.

The Release workflow requires successful pre-release validation of the tagged commit before
publishing to PyPI or updating docs. Any commit change, including a version bump, requires a new
validation run. Publishing the GitHub release itself does not bypass this check.

## Need Help?

- Check existing issues and PRs
- Look at similar code in the repository
- Ask questions in your PR

> By contributing, you agree that your contributions will be licensed under the MIT License.
