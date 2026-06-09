"""Smart HUD environment initialization."""

from __future__ import annotations

import subprocess
from pathlib import Path

import questionary
import typer

from hud.utils.hud_console import HUDConsole

from .templates import DOCKERFILE_HUD, ENV_PY, PYPROJECT_TOML, TASKS_PY

# Files that indicate this might be an existing project
PROJECT_INDICATORS = {
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "setup.py",
    "Cargo.toml",
    "go.mod",
}


def _normalize_name(name: str) -> str:
    """Normalize name for Python identifiers."""
    name = name.replace("-", "_").replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _has_hud_dependency(directory: Path) -> bool:
    """Check if hud-python is already in pyproject.toml."""
    pyproject = directory / "pyproject.toml"
    if not pyproject.exists():
        return False
    content = pyproject.read_text()
    return "hud-python" in content or "hud_python" in content


def _add_hud_dependency(directory: Path) -> str:
    """Add hud-python using uv if available.

    Returns:
        "exists" if already present, "added" if added, "failed" if failed
    """
    if _has_hud_dependency(directory):
        return "exists"

    try:
        result = subprocess.run(
            ["uv", "add", "hud-python", "openai"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=directory,
            check=False,
        )
        if result.returncode == 0 or "already" in result.stderr.lower():
            return "added"
        return "failed"
    except FileNotFoundError:
        return "failed"


def _is_empty_or_trivial(directory: Path) -> bool:
    """Check if directory is empty or only has trivial files."""
    if not directory.exists():
        return True
    files = list(directory.iterdir())
    if not files:
        return True
    trivial = {".git", ".gitignore", ".DS_Store", "README.md", "LICENSE"}
    return all(f.name in trivial or f.name.startswith(".") for f in files)


def _has_project_files(directory: Path) -> bool:
    """Check if directory has files indicating an existing project."""
    if not directory.exists():
        return False
    return any(f.name in PROJECT_INDICATORS for f in directory.iterdir())


def _prompt_init_mode(target: Path) -> str | None:
    """Ask the user whether to init inside the current directory or create a new one.

    Returns "here", "new", or None if cancelled.
    """
    try:
        selected = questionary.select(
            f"Directory '{target.name}' already contains files. How would you like to initialize?",
            choices=[
                questionary.Choice(
                    "Add HUD files to this directory",
                    value="here",
                ),
                questionary.Choice(
                    "Create a new environment in a subdirectory (from preset)",
                    value="new",
                ),
            ],
        ).ask()
        return selected
    except KeyboardInterrupt:
        return None


def _init_in_existing_directory(
    target: Path,
    name: str | None,
    force: bool,
) -> None:
    """Add HUD files to an existing project directory."""
    hud_console = HUDConsole()

    target.mkdir(parents=True, exist_ok=True)
    env_name = _normalize_name(name or target.name)
    has_pyproject = (target / "pyproject.toml").exists()

    hud_console.header(f"HUD Init: {env_name}")

    if has_pyproject:
        hud_console.info("Found pyproject.toml - adding HUD files")
    else:
        hud_console.info("Creating HUD environment in existing directory")

    created: list[str] = []

    if not has_pyproject:
        pyproject = target / "pyproject.toml"
        pyproject.write_text(PYPROJECT_TOML.format(name=env_name.replace("_", "-")))
        created.append("pyproject.toml")

    dockerfile = target / "Dockerfile.hud"
    if not dockerfile.exists() or force:
        dockerfile.write_text(DOCKERFILE_HUD)
        created.append("Dockerfile.hud")
    else:
        hud_console.warning("Dockerfile.hud exists, skipping (use --force)")

    env_py = target / "env.py"
    if not env_py.exists() or force:
        env_py.write_text(ENV_PY.format(env_name=env_name))
        created.append("env.py")
    else:
        hud_console.warning("env.py exists, skipping (use --force)")

    tasks_py = target / "tasks.py"
    if not tasks_py.exists() or force:
        tasks_py.write_text(TASKS_PY.format(env_name=env_name))
        created.append("tasks.py")
    else:
        hud_console.warning("tasks.py exists, skipping (use --force)")

    dep_result = _add_hud_dependency(target)
    if dep_result == "added":
        hud_console.success("Added hud-python dependency")
    elif dep_result == "exists":
        hud_console.info("hud-python already in dependencies")
    else:
        hud_console.info("Run manually: uv add hud-python openai")

    if created:
        hud_console.section_title("Created")
        for f in created:
            hud_console.status_item(f, "✓")

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.info("1. Define tasks in env.py")
    hud_console.info("   A @env.task is an async generator: it yields a prompt, then")
    hud_console.info("   (after the agent answers) yields a reward.")
    hud_console.info("")
    hud_console.info("2. List the tasks to run in tasks.py")
    hud_console.info("   Call a task with args to bind a runnable Task.")
    hud_console.info("")
    hud_console.info("3. Run an agent over them")
    hud_console.command_example("hud eval tasks.py claude", "Evaluate locally")
    hud_console.info("")
    hud_console.info("4. Deploy for scale")
    hud_console.info("   hud build, hud deploy, then run many evals in parallel.")
    hud_console.info("")
    hud_console.section_title("Files")
    hud_console.info("• env.py         Your environment: capabilities + @env.task tasks")
    hud_console.info("• tasks.py       The Tasks to evaluate (hud eval tasks.py <agent>)")
    hud_console.info("• Dockerfile.hud Container config for deployment")


def smart_init(
    name: str | None = None,
    directory: str = ".",
    force: bool = False,
) -> None:
    """Initialize HUD environment, always prompting the user for what to do."""
    from hud.settings import settings

    hud_console = HUDConsole()

    if not settings.api_key:
        hud_console.error("HUD_API_KEY not found")
        hud_console.info("")
        hud_console.info("Set your API key:")
        hud_console.info("  hud set HUD_API_KEY=your-key-here")
        hud_console.info("  Or: export HUD_API_KEY=your-key")
        hud_console.info("")
        hud_console.info("Get your key at: https://hud.ai/project/api-keys")
        return

    target = Path(directory).resolve()

    if _is_empty_or_trivial(target):
        from hud.cli.init import create_environment

        create_environment(name, directory, force, preset=None)
        return

    # Non-empty directory — ask the user what they want
    mode = _prompt_init_mode(target)

    if mode is None:
        raise typer.Exit(0)

    if mode == "here":
        _init_in_existing_directory(target, name, force)
    else:
        from hud.cli.init import create_environment

        create_environment(name, directory, force, preset=None)


__all__ = ["smart_init"]
