"""Initialize new HUD environments with minimal templates."""

from __future__ import annotations

import os
import tarfile
import tempfile
import time
from pathlib import Path

import httpx
import questionary
import typer

from hud.utils.hud_console import HUDConsole

# Presets mapping to public GitHub repositories under hud-evals org
GITHUB_OWNER = "hud-evals"
GITHUB_BRANCH = "main"

PRESET_MAP: dict[str, str | None] = {
    "blank": "hud-blank",
    "deep-research": "hud-deepresearch",
    "browser": "hud-browser",
    "rubrics": "hud-rubrics",
}

SKIP_DIR_NAMES = {"node_modules", "__pycache__", "dist", "build", ".next", ".git"}

# Files that need placeholder replacement
PLACEHOLDER_FILES = {
    "server/pyproject.toml",
    "environment/pyproject.toml",
    "server/main.py",
    "server/README.md",
    "environment/README.md",
    "tasks.json",
    "test_env.ipynb",
    "README.md",
}


def _replace_placeholders(target_dir: Path, env_name: str) -> list[str]:
    """Replace placeholders in template files with the actual environment name.

    Args:
        target_dir: Directory containing the downloaded template files
        env_name: The environment name to replace placeholders with

    Returns:
        List of files that were modified
    """
    modified_files = []
    placeholder = "blank"  # Placeholder used in blank environment template

    # Normalize environment name for use in code/configs
    # Replace spaces and special chars with underscores for Python identifiers
    normalized_name = env_name.replace("-", "_").replace(" ", "_")
    normalized_name = "".join(c if c.isalnum() or c == "_" else "_" for c in normalized_name)

    for root, dirs, files in os.walk(target_dir):
        # Skip directories we don't want to process
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]

        for file in files:
            file_path = Path(root) / file

            # Check if this file should have placeholders replaced
            should_replace = file in PLACEHOLDER_FILES or any(
                file_path.relative_to(target_dir).as_posix().endswith(f) for f in PLACEHOLDER_FILES
            )

            if should_replace:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if placeholder in content:
                        new_content = content.replace(placeholder, normalized_name)
                        file_path.write_text(new_content, encoding="utf-8")
                        modified_files.append(str(file_path.relative_to(target_dir)))
                except Exception:  # noqa: S110
                    # Skip files that can't be read as text
                    pass

    return modified_files


def _prompt_for_preset() -> str:
    """Ask the user to choose a preset when not provided."""
    try:
        choices = [
            {"name": "blank", "message": "blank"},
            {"name": "browser", "message": "browser"},
            {"name": "deep-research", "message": "deep-research"},
            {"name": "rubrics", "message": "rubrics"},
        ]
        display_choices = [c["message"] for c in choices]
        selected = questionary.select(
            "Choose a preset", choices=display_choices, default=display_choices[0]
        ).ask()
        if not selected:
            return "blank"
        for c in choices:
            if c["message"] == selected:
                return c["name"]
        return "blank"
    except Exception:
        return "blank"


def _download_tarball_repo(
    owner: str, repo: str, ref: str, dest_dir: Path, files_created: list[str]
) -> None:
    """Download a GitHub tarball and extract the entire repository."""
    tarball_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    with (
        tempfile.NamedTemporaryFile(delete=False) as tmp_file,
        httpx.Client(timeout=60) as client,
        client.stream(
            "GET",
            tarball_url,
            headers=headers,
        ) as resp,
    ):
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download tarball (HTTP {resp.status_code}) from {tarball_url}"
            )
        for chunk in resp.iter_bytes():
            if chunk:
                tmp_file.write(chunk)
        tmp_path = Path(tmp_file.name)

    try:
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            members = tar.getmembers()
            if not members:
                return
            top = members[0].name.split("/", 1)[0]

            for member in members:
                name = member.name
                if name == top:
                    continue

                if not name.startswith(top + "/"):
                    continue

                rel_path = name[len(top) + 1 :]
                if not rel_path:
                    continue

                out_path = (dest_dir / rel_path).resolve()
                dest_root = dest_dir.resolve()
                if not str(out_path).startswith(str(dest_root)):
                    continue

                if member.isdir():
                    out_path.mkdir(parents=True, exist_ok=True)
                elif member.isreg():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    with open(out_path, "wb") as f:
                        f.write(extracted.read())
                    # Use absolute dest_root for relative path computation to avoid Windows issues
                    files_created.append(str(out_path.relative_to(dest_root)))
    finally:
        from contextlib import suppress

        with suppress(Exception):
            os.remove(tmp_path)


def create_environment(
    name: str | None, directory: str, force: bool, preset: str | None = None
) -> None:
    """Create a new HUD environment by downloading a preset from the repo."""

    hud_console = HUDConsole()

    # Choose preset
    preset_normalized = (preset or "").strip().lower() if preset else _prompt_for_preset()

    # If no name is provided, use the preset name as the environment name
    if name is None:
        name = preset_normalized
        hud_console.info(f"Using preset name as environment name: {name}")

    # Always create a new directory based on the name
    target_dir = Path.cwd() / name if directory == "." else Path(directory) / name

    if preset_normalized not in PRESET_MAP:
        available = ", ".join(sorted(PRESET_MAP.keys()))
        hud_console.warning(
            f"Unknown preset '{preset_normalized}', defaulting to 'blank' (available: {available})"
        )
        preset_normalized = "blank"

    # Check if directory exists
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            hud_console.error(f"Directory {target_dir} already exists and is not empty")
            hud_console.info("Use --force to overwrite existing files")
            raise typer.Exit(1)
        else:
            hud_console.warning(f"Overwriting existing files in {target_dir}")

    # Download preset from GitHub
    repo_name = PRESET_MAP[preset_normalized]
    if repo_name is None:
        hud_console.error("Internal error: preset mapping missing repo name")
        raise typer.Exit(1)

    hud_console.header(f"Initializing HUD Environment: {name} (preset: {preset_normalized})")
    hud_console.section_title("Downloading template from GitHub")
    source_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}"
    hud_console.info("Source: " + source_url)

    target_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    files_created_dl: list[str] = []
    try:
        _download_tarball_repo(
            owner=GITHUB_OWNER,
            repo=repo_name,
            ref=GITHUB_BRANCH,
            dest_dir=target_dir,
            files_created=files_created_dl,
        )
    except Exception as e:
        hud_console.error(f"Failed to download preset '{preset_normalized}': {e}")
        raise typer.Exit(1) from None

    duration_ms = int((time.time() - started) * 1000)
    hud_console.success(
        f"Downloaded {len(files_created_dl)} files in {duration_ms} ms into {target_dir}"
    )

    # Replace placeholders in template files (only for blank preset)
    if preset_normalized == "blank":
        hud_console.section_title("Customizing template files")
        modified_files = _replace_placeholders(target_dir, name)
        if modified_files:
            hud_console.success(f"Replaced placeholders in {len(modified_files)} files:")
            for file in modified_files[:5]:  # Show first 5 files
                hud_console.status_item(file, "updated")
            if len(modified_files) > 5:
                hud_console.info(f"... and {len(modified_files) - 5} more files")
        else:
            hud_console.info("No placeholder replacements needed")

    hud_console.section_title("Top-level files and folders")
    for entry in sorted(os.listdir(target_dir)):
        hud_console.status_item(entry, "added")

    hud_console.section_title("Next steps")
    # Since we now almost always create a new directory, show cd command
    hud_console.info("1. Enter the directory:")
    hud_console.command_example(f"cd {target_dir.name}")
    hud_console.info("\n2. Start development server (with MCP inspector):")
    hud_console.command_example("hud dev --inspector")
    hud_console.info("\n3. Review the README in this preset for specific instructions.")
    hud_console.info("\n4. Customize as needed.")
