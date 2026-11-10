# Contributing to HUD

The most useful contributions come from using HUD: you were building an environment, running an
eval, or training a model, and something got in the way.

## Issues

We act on problems that happened to you while using HUD. A problem that could affect users in
theory is not enough; tell us how it affected you.

Open a **Bug report** when a run went wrong, or a **Gap** when HUD couldn't do what you needed.
Describe what you were doing and what the problem cost you. If you can, link a public trace of the
run: it shows us exactly what happened, and reports with one are much more likely to be acted on.
When we decide to fix an issue, we label it `actionable`.

## Pull requests

Outside pull requests must fix an `actionable` issue, or they are closed automatically. Put
`Fixes #<issue>` in the description, keep the change to that issue, and add a test that fails
without the fix when the fix could regress (see [Writing Tests](#writing-tests)).

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/hud-python`
3. Install [uv](https://docs.astral.sh/uv/) and set up dev dependencies:
   ```bash
   cd hud-python
   uv sync --all-extras
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
uv run ty check                # Type checking
```

## Code Style

- Python 3.11+ features are allowed
- Type hints required for public APIs
- Line length limit: 100 characters
- Follow existing patterns in the codebase

## Need Help?

Questions about using HUD go to [Discord](https://discord.gg/wkjtmHYYjm). The
[internals docs](https://docs.hud.ai/v6/internals/walkthrough) follow one full run through the code.

> By contributing, you agree that your contributions will be licensed under the MIT License.
