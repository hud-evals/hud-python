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
what lies outside it.

- `tests/support`: stand-ins for the outside world (HUD services, model
  providers, a browser, a VNC screen, the network, the user's machine).
- `tests/fixtures`: SDK code the tests run: fixture environments and an agent with
  fixed answers.
- `tests/conftest.py`: the fixtures that wire them in: `hud_env` (configuration),
  `platform` and `services` (HUD services), `models` (scripted providers) and
  `hud` (the CLI in a subprocess).

```python
async def test_a_graded_rollout_reports_its_reward(reporting):
    job = await Task(env="echo", id="echo", args={"word": "hi"}).run(
        ScriptedAgent("hi"), runtime=LocalRuntime(source("echo"))
    )

    (trace,) = reporting.traces.values()
    assert (job.reward, trace.exit["reward"]) == (1.0, 1.0)
```

Test directories follow the SDK's subsystems (`environment`, `clients`, `eval`,
`agents`, `telemetry`, `cli`, ...), and each file holds the contracts of one owner.
When a contract has several implementations (runtimes, agent families, commands,
capability clients), they are rows of one table, and a completeness test fails
when an implementation has no row.

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
