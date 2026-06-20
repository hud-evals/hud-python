"""Starter presets for ``hud init`` — the same set offered by the platform's
*environments/new* flow.

Each preset is a standalone public GitHub repo under ``hud-evals``. ``hud init``
downloads the repo tarball (no ``git`` required) and extracts it into the target
directory. Keep this list in sync with the frontend's ``ENVIRONMENT_TEMPLATES``
(``app/(auth)/environments/components/EnvironmentTemplates.tsx``).
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class EnvironmentPreset:
    """A starter environment sourced from a public GitHub repo."""

    id: str
    name: str
    description: str
    owner: str
    repo: str


ENVIRONMENT_PRESETS: tuple[EnvironmentPreset, ...] = (
    EnvironmentPreset(
        "blank",
        "Blank",
        "Minimal starting point for building your own environment.",
        "hud-evals",
        "hud-blank",
    ),
    EnvironmentPreset(
        "browser",
        "Browser",
        "Browser agents: a 2048 game and a todo app in real Chromium (cdp + rfb).",
        "hud-evals",
        "hud-browser",
    ),
    EnvironmentPreset(
        "cua",
        "Computer Use",
        "Computer-use agents: a virtual Linux desktop (XFCE + Chromium) over rfb/VNC.",
        "hud-evals",
        "cua-template",
    ),
    EnvironmentPreset(
        "deepresearch",
        "Deep Research",
        "Live deep research: web search (Exa) and People/Company search (SixtyFour).",
        "hud-evals",
        "hud-deepresearch",
    ),
    EnvironmentPreset(
        "coding",
        "Coding",
        "Fix a bug in a Python web app, graded by a hidden pytest suite.",
        "hud-evals",
        "coding-template",
    ),
    EnvironmentPreset(
        "ml",
        "ML Research/Training",
        "ML research and training tasks (GPU).",
        "hud-evals",
        "ml-template",
    ),
    EnvironmentPreset(
        "ml-triage",
        "ML Triage/Productivity",
        "ML triage and productivity tasks.",
        "hud-evals",
        "ml-triage-tasks",
    ),
    EnvironmentPreset(
        "verilog",
        "Verilog",
        "Chip design: one Verilog/SystemVerilog task over ssh, graded by hidden EDA flows.",
        "hud-evals",
        "verilog-template",
    ),
    EnvironmentPreset(
        "autonomous-businesses",
        "Autonomous Businesses",
        "Autonomous business loop: support-ticket triage for a small clinic.",
        "hud-evals",
        "autonomous-businesses-template",
    ),
    EnvironmentPreset(
        "gdpval",
        "GDPVal",
        "GDPVal benchmark tasks.",
        "hud-evals",
        "gdpval-template",
    ),
    EnvironmentPreset(
        "worldsim",
        "Worldsim",
        "Robotics: a Newton physics scene driven by an LLM agent or VLA policy (AntimLabs).",
        "hud-evals",
        "worldsim-template",
    ),
    EnvironmentPreset(
        "videogamebench",
        "VideoGameBench",
        "Evaluate agents on classic Game Boy games (AntimLabs).",
        "hud-evals",
        "videogamebench-template",
    ),
)

PRESETS_BY_ID: dict[str, EnvironmentPreset] = {p.id: p for p in ENVIRONMENT_PRESETS}

_TARBALL_TIMEOUT = 60.0


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _download_tarball(preset: EnvironmentPreset) -> bytes:
    """Fetch the repo's ``main`` archive from codeload (no API rate limit)."""
    headers: dict[str, str] = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://codeload.github.com/{preset.owner}/{preset.repo}/tar.gz/refs/heads/main"
    with httpx.Client(follow_redirects=True, timeout=_TARBALL_TIMEOUT) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


def materialize_preset(preset: EnvironmentPreset, target: Path) -> None:
    """Download ``preset``'s repo archive and extract it into ``target``.

    Uses ``codeload.github.com`` (not the rate-limited API) for the repo's
    ``main`` branch — no ``git`` required. Strips the archive's top-level
    ``<repo>-main/`` component and refuses any entry that would escape ``target``
    (path-traversal guard). Honors ``GITHUB_TOKEN`` if set.
    """
    payload = _download_tarball(preset)

    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            # GitHub wraps everything in a "<repo>-<sha>/" top-level dir; drop it.
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            dest = (target_root / parts[1]).resolve()
            if not _is_within(target_root, dest):
                raise ValueError(f"unsafe path in archive: {member.name!r}")
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is not None:
                    dest.write_bytes(source.read())
                    # Preserve the archive's executable bits so entrypoints and
                    # scripts stay runnable (no-op on Windows).
                    if member.mode & 0o111:
                        dest.chmod(dest.stat().st_mode | (member.mode & 0o111))
            # Symlinks and other special members are intentionally skipped.
