"""Smart HUD environment initialization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hud.utils.hud_console import HUDConsole

from .templates import DOCKERFILE_HUD, ENV_PY, PYPROJECT_TOML

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
    # Empty
    if not files:
        return True
    # Only has hidden files or common trivial files
    trivial = {".git", ".gitignore", ".DS_Store", "README.md", "LICENSE"}
    return all(f.name in trivial or f.name.startswith(".") for f in files)


def _has_project_files(directory: Path) -> bool:
    """Check if directory has files indicating an existing project."""
    if not directory.exists():
        return False
    return any(f.name in PROJECT_INDICATORS for f in directory.iterdir())


def smart_init(
    name: str | None = None,
    directory: str = ".",
    force: bool = False,
) -> None:
    """Initialize HUD environment files in a directory.

    - If directory is empty: delegate to preset selection
    - If directory has project files: add HUD files to existing project
    - Otherwise: create new HUD environment
    """
    from hud.settings import settings

    hud_console = HUDConsole()

    # Check for API key first
    if not settings.api_key:
        hud_console.error("HUD_API_KEY not found")
        hud_console.info("")
        hud_console.info("Set your API key:")
        hud_console.info("  hud set HUD_API_KEY=your-key-here")
        hud_console.info("  Or: export HUD_API_KEY=your-key")
        hud_console.info("")
        hud_console.info("Get your key at: https://hud.ai/settings/api-keys")
        return

    target = Path(directory).resolve()

    # If directory is empty, use preset selection
    if _is_empty_or_trivial(target):
        from hud.cli.init import create_environment

        hud_console.info("Empty directory - showing preset selection")
        create_environment(name, directory, force, preset=None)
        return

    # Directory has files - use smart init
    target.mkdir(parents=True, exist_ok=True)
    env_name = _normalize_name(name or target.name)
    has_pyproject = (target / "pyproject.toml").exists()

    hud_console.header(f"HUD Init: {env_name}")

    if has_pyproject:
        hud_console.info("Found pyproject.toml - adding HUD files")
    else:
        hud_console.info("Creating HUD environment in existing directory")

    created = []

    # Create pyproject.toml if needed
    if not has_pyproject:
        pyproject = target / "pyproject.toml"
        pyproject.write_text(PYPROJECT_TOML.format(name=env_name.replace("_", "-")))
        created.append("pyproject.toml")

    # Create Dockerfile.hud
    dockerfile = target / "Dockerfile.hud"
    if not dockerfile.exists() or force:
        dockerfile.write_text(DOCKERFILE_HUD)
        created.append("Dockerfile.hud")
    else:
        hud_console.warning("Dockerfile.hud exists, skipping (use --force)")

    # Create env.py
    env_py = target / "env.py"
    if not env_py.exists() or force:
        env_py.write_text(ENV_PY.format(env_name=env_name))
        created.append("env.py")
    else:
        hud_console.warning("env.py exists, skipping (use --force)")

    # Add dependency
    dep_result = _add_hud_dependency(target)
    if dep_result == "added":
        hud_console.success("Added hud-python dependency")
    elif dep_result == "exists":
        hud_console.info("hud-python already in dependencies")
    else:
        hud_console.info("Run manually: uv add hud-python openai")

    # Summary
    if created:
        hud_console.section_title("Created")
        for f in created:
            hud_console.status_item(f, "✓")

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.info("1. Define your tools in env.py")
    hud_console.info("   Tools are functions the agent can call. Wrap existing code")
    hud_console.info("   with @env.tool() or connect FastAPI/OpenAPI servers.")
    hud_console.info("")
    hud_console.info("2. Write scripts that test agent behavior")
    hud_console.info("   Scripts define prompts and scoring. The agent runs between")
    hud_console.info("   two yields: first sends the task, second scores the result.")
    hud_console.info("")
    hud_console.info("3. Run locally to iterate")
    hud_console.command_example("python env.py", "Run the test script")
    hud_console.info("")
    hud_console.info("4. Deploy for scale")
    hud_console.info("   Push to GitHub, connect on hud.ai. Then run hundreds of")
    hud_console.info("   evals in parallel and collect training data.")
    hud_console.info("")
    hud_console.section_title("Files")
    hud_console.info("• env.py         Your tools, scripts, and test code")
    hud_console.info("• Dockerfile.hud Container config for remote deployment")


__all__ = ["smart_init"]
