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
uv run pytest -m e2e tests/e2e/test_cookbooks.py   # the cookbooks; CI checks them nightly
```

### Writing Tests

Tests are black-box: they drive the SDK through a public boundary and fake only
what lies outside it. `tests/harness` holds the fakes, and `tests/conftest.py`
provides them as fixtures:

- `hud_env`: the SDK's configuration. Every test starts with a temporary home,
  no credentials, and service URLs pointed at a closed port; `hud_env.set(...)`
  changes environment variables the way a user would.
- `services`: a local stand-in for the HUD API, telemetry, gateway, runtime and
  training services. Declare replies with `services.route(...)` and read what
  the SDK sent with `services.requests(...)`; anything undeclared answers 404.
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
async def test_a_reporting_rollout_reaches_the_platform(services, hud_env):
    hud_env.set(HUD_API_KEY="k", HUD_TELEMETRY_ENABLED="1")
    services.route("api", "POST", "/v2/trace/{id}/exit", json={})

    job = await Taskset("demo", [row]).run(ScriptedAgent("3"), runtime=LocalRuntime(env))

    assert job.reward == 1.0
    assert services.bodies("api", "POST", "/v2/trace/{id}/exit") == snapshot(...)
```

Rules, enforced by `scripts/check_tests.py` and in review:

- Never patch a `hud` name, import a private one, or touch a private attribute.
- Never hand-build what a producer emits; run the producer.
- Write one scenario per contract as a table, and add a row for a new case.
- Use inline snapshots (`inline-snapshot`) only where a whole output is the
  contract. Record them with `uv run pytest --inline-snapshot=create` and review
  every change to one like any other diff.
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
