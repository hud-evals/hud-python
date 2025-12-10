"""Smart HUD environment initialization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hud.utils.hud_console import HUDConsole

from .templates import DOCKERFILE_HUD, HUD_PY, PYPROJECT_TOML

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


def _add_hud_dependency(directory: Path) -> bool:
    """Add hud-python using uv if available."""
    try:
        result = subprocess.run(
            ["uv", "add", "hud-python", "openai"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=directory,
            check=False,
        )
        return result.returncode == 0 or "already" in result.stderr.lower()
    except FileNotFoundError:
        return False


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
    hud_console = HUDConsole()
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

    # Create hud.py
    hud_py = target / "hud.py"
    if not hud_py.exists() or force:
        hud_py.write_text(HUD_PY.format(env_name=env_name))
        created.append("hud.py")
    else:
        hud_console.warning("hud.py exists, skipping (use --force)")

    # Add dependency
    if _add_hud_dependency(target):
        hud_console.success("Added hud-python dependency")
    else:
        hud_console.info("Run manually: uv add hud-python openai")

    # Summary
    if created:
        hud_console.section_title("Created")
        for f in created:
            hud_console.status_item(f, "✓")

    hud_console.section_title("Next Steps")
    hud_console.info("1. Edit hud.py:")
    hud_console.info("   - Add your tools with @env.tool()")
    hud_console.info("   - Connect existing servers (FastAPI, MCP, OpenAPI)")
    hud_console.info("")
    hud_console.info("2. Edit Dockerfile.hud:")
    hud_console.info("   - Add system dependencies (apt-get install)")
    hud_console.info("   - Set up data sources for production")
    hud_console.info("")
    hud_console.command_example("python hud.py", "Test locally")
    hud_console.command_example("hud dev hud:env", "Development server")
    hud_console.command_example("hud build", "Build Docker image")
    hud_console.info("")
    hud_console.section_title("Tips")
    hud_console.info("• For production environments you want to mock locally,")
    hud_console.info("  configure data sources in Dockerfile.hud before deploying")
    hud_console.info("• For testing without real connections, use env.mock()")
    hud_console.info("• See hud.py DEPLOYMENT section for remote deployment")


__all__ = ["smart_init"]
