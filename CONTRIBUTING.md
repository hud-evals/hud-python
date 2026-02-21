# Contributing to HUD

We welcome contributions to the HUD SDK! This guide covers how to get started.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/hud-python`
3. Install [uv](https://docs.astral.sh/uv/) and set up dev dependencies:
   ```bash
   cd hud-python
   uv sync --extra dev
   ```

## Development Workflow

### Git Hooks

Enable the shared pre-push hook (runs ruff, pyright, pytest before each push):

```bash
git config core.hooksPath .githooks
```

### Running Tests

```bash
uv run pytest --rootdir=hud -q
```

Tests run on Python 3.11 and 3.12 in CI.

### Code Quality

```bash
uv run ruff format . --check   # Formatting
uv run ruff check .            # Linting
uv run pyright                 # Type checking
```

## Code Style

- Python 3.11+ features are allowed
- Type hints required for public APIs
- Line length limit: 100 characters
- Follow existing patterns in the codebase

## Pull Request Process

1. **Branch naming**: `feature/description` or `fix/issue-number`
2. **Commits**: Use clear, descriptive messages
3. **Tests**: All CI checks must pass (ruff, pyright, pytest)
4. **Review**: Address feedback promptly

## Need Help?

- Check existing issues and PRs
- Look at similar code in the repository
- Ask questions in your PR

> By contributing, you agree that your contributions will be licensed under the MIT License.
