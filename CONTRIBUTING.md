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
uv run --with "lerobot[dataset]" python -m pytest -m "e2e and lerobot"  # real LeRobot datasets
uv run pytest -m e2e tests/e2e/test_cookbooks.py   # the cookbooks
```

### Writing Tests

Write a test only when it would catch a break a user would notice; a change
needs none by default. Drive the SDK as a user does and assert what they see:
reward, status, errors, files, exit codes, and what the HUD platform records.
Fake only what CI cannot run; the fakes and fixture environments live in
`tests/fixtures`, and `tests/conftest.py` wires them in (`services`, `models`,
`hud`, `hud_env`). A regression test must fail with its fix reverted.

```python
env = Environment("echo")


@env.template()
async def echo(word: str):
    answer = yield f"Say {word}."
    yield 1.0 if answer == word else 0.0


async def test_a_graded_rollout_reports_its_reward(reporting):
    job = await echo(word="hi").run(ScriptedAgent("hi"))

    (exited,) = reporting.bodies("api", "POST", "/v2/trace/{id}/exit")
    assert (job.reward, exited["reward"]) == (1.0, 1.0)
```

### Code Quality

```bash
uv run ruff format . --check   # Formatting
uv run ruff check .            # Linting
uv run --extra dev --extra train ty check  # Type checking
```

## Code Style

- Python 3.11+ features are allowed
- Type hints required for public APIs
- Line length limit: 100 characters
- Follow existing patterns in the codebase

## Pull Request Process

Pull requests from outside the maintainer team must fix an issue a maintainer has
labelled `accepted`; others are closed automatically. Open an issue describing the
change (or find an existing one), wait for the label, then put `Fixes #<issue>` in
your pull request description.

1. **Branch naming**: `feature/description` or `fix/issue-number`
2. **Commits**: Use clear, descriptive messages
3. **Checks**: All CI checks must pass (ruff, ty, pytest)
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
