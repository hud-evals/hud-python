"""Skill injection helpers for loading markdown files into agent context.

Skills are markdown files that provide domain-specific instructions,
workflows, or knowledge to agents. This module provides helpers to
load them into system prompts or scenario context.

Usage::

    from hud.native.skills import load_skills

    # Load individual files
    agent = ClaudeAgent.create(system_prompt=load_skills("skills/code_review.md", "skills/git.md"))

    # Load entire directory
    agent = ClaudeAgent.create(system_prompt=load_skills("skills/"))


    # In a scenario
    @env.scenario()
    async def review(pr_url: str):
        skills = load_skills("skills/review.md")
        yield f"{skills}\\n\\n---\\n\\nReview this PR: {pr_url}"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def load_skills(*paths: str | Path, separator: str = "\n\n---\n\n") -> str:
    """Load markdown skill files and format as agent context.

    Each file becomes a section with its filename (without extension) as
    the heading. Directories are expanded to all ``.md`` files sorted
    alphabetically.

    Args:
        *paths: File or directory paths to load. Directories are
            expanded to all ``*.md`` files within them.
        separator: String used between sections.

    Returns:
        Concatenated skill content ready for system prompt injection.

    Raises:
        FileNotFoundError: If a path does not exist.

    Example::

        # Single file
        ctx = load_skills("skills/git.md")

        # Multiple files
        ctx = load_skills("skills/git.md", "skills/review.md")

        # Directory (loads all .md files alphabetically)
        ctx = load_skills("skills/")

        # Mixed
        ctx = load_skills("skills/", "extra/custom.md")
    """
    sections: list[str] = []

    for raw_path in paths:
        p = Path(raw_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Skill path not found: {p}")

        if p.is_dir():
            md_files = sorted(p.glob("*.md"))
            if not md_files:
                LOGGER.warning("No .md files found in skill directory: %s", p)
                continue
            sections.extend(_load_one(md) for md in md_files)
        else:
            sections.append(_load_one(p))

    return separator.join(sections)


def load_skills_from_config(
    config: dict[str, Any],
    key: str = "skills",
    base_path: str | Path | None = None,
) -> str | None:
    """Load skills from a configuration dict.

    Useful for loading skills specified in task configs or environment
    settings.

    Args:
        config: Configuration dict containing skill paths.
        key: Key in config that holds skill path(s).
        base_path: Base directory for resolving relative paths.

    Returns:
        Loaded skill content, or None if key is not present.
    """
    raw = config.get(key)
    if raw is None:
        return None

    paths: list[str | Path]
    if isinstance(raw, str):
        paths = [raw]
    elif isinstance(raw, list):
        paths = raw
    else:
        LOGGER.warning("Invalid skills config value (expected str or list): %s", type(raw))
        return None

    if base_path is not None:
        base = Path(base_path)
        paths = [base / p if not Path(p).is_absolute() else p for p in paths]

    return load_skills(*paths)


def _load_one(path: Path) -> str:
    """Load a single markdown file as a skill section."""
    title = path.stem.replace("_", " ").replace("-", " ").title()
    content = path.read_text(encoding="utf-8").strip()
    return f"## {title}\n\n{content}"
