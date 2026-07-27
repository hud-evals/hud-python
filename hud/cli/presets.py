"""Starter presets for ``hud init`` — the same set offered by the platform's
*environments/new* flow.

Vendored presets ship inside the wheel, so the starter a user scaffolds always
matches the SDK version they installed. The rest are standalone public GitHub
repos under ``hud-evals``: ``hud init`` downloads the repo tarball (no ``git``
required) and extracts it into the target directory. Keep this list in sync with
the frontend's ``ENVIRONMENT_TEMPLATES``
(``app/(auth)/environments/components/EnvironmentTemplates.tsx``).
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True, slots=True)
class EnvironmentPreset:
    """A starter environment.

    ``vendored`` presets are copied from the tree bundled with this package;
    the rest are downloaded from ``owner``/``repo``. ``repo`` also names the
    target directory when ``hud init`` is run without a name.
    """

    id: str
    emoji: str
    name: str
    description: str
    owner: str
    repo: str
    vendored: bool = False


ENVIRONMENT_PRESETS: tuple[EnvironmentPreset, ...] = (
    EnvironmentPreset(
        "blank",
        "🧱",
        "Blank",
        "Minimal bundled scaffold (no download): a single letter-counting task.",
        "hud-evals",
        "hud-blank",
        vendored=True,
    ),
    EnvironmentPreset(
        "browser",
        "🌐",
        "Browser",
        "Browser agents: a 2048 game and a todo app in real Chromium (cdp + rfb).",
        "hud-evals",
        "hud-browser",
    ),
    EnvironmentPreset(
        "cua",
        "🖥️",
        "Computer Use",
        "Computer-use agents: a virtual Linux desktop (XFCE + Chromium) over rfb/VNC.",
        "hud-evals",
        "cua-template",
        vendored=True,
    ),
    EnvironmentPreset(
        "deepresearch",
        "🔬",
        "Deep Research",
        "Live deep research: web search (Exa) and People/Company search (SixtyFour).",
        "hud-evals",
        "hud-deepresearch",
    ),
    EnvironmentPreset(
        "coding",
        "🐛",
        "Coding",
        "Fix a bug in a Python web app, graded by a hidden pytest suite.",
        "hud-evals",
        "coding-template",
        vendored=True,
    ),
    EnvironmentPreset(
        "ml",
        "🧠",
        "ML Research/Training",
        "ML research and training tasks (GPU).",
        "hud-evals",
        "ml-template",
    ),
    EnvironmentPreset(
        "ml-triage",
        "🩺",
        "ML Triage/Productivity",
        "ML triage and productivity tasks.",
        "hud-evals",
        "ml-triage-tasks",
    ),
    EnvironmentPreset(
        "verilog",
        "🔌",
        "Verilog",
        "Chip design: one Verilog/SystemVerilog task over ssh, graded by hidden EDA flows.",
        "hud-evals",
        "verilog-template",
    ),
    EnvironmentPreset(
        "autonomous-businesses",
        "💼",
        "Autonomous Businesses",
        "Autonomous business loop: support-ticket triage for a small clinic.",
        "hud-evals",
        "autonomous-businesses-template",
    ),
    EnvironmentPreset(
        "gdpval",
        "📈",
        "GDPVal",
        "GDPVal benchmark tasks.",
        "hud-evals",
        "gdpval-template",
    ),
    EnvironmentPreset(
        "worldsim",
        "🦾",
        "Worldsim",
        "Robotics: a Newton physics scene driven by an LLM agent or VLA policy (AntimLabs).",
        "hud-evals",
        "worldsim-template",
    ),
    EnvironmentPreset(
        "robot",
        "🤖",
        "Robot",
        "Robotics: run a VLA policy against a containerized robot sim, graded by task success.",
        "hud-evals",
        "robot-template",
    ),
    EnvironmentPreset(
        "videogamebench",
        "🎮",
        "VideoGameBench",
        "Evaluate agents on classic Game Boy games (AntimLabs).",
        "hud-evals",
        "videogamebench-template",
    ),
    EnvironmentPreset(
        "arc-agi-3",
        "🧩",
        "ARC-AGI-3",
        "Interactive reasoning benchmark: agents play ARC-AGI-3 games.",
        "hud-evals",
        "ARC-AGI-3",
    ),
)

PRESETS_BY_ID: dict[str, EnvironmentPreset] = {p.id: p for p in ENVIRONMENT_PRESETS}

_TARBALL_TIMEOUT = 60.0

# Installed layout: the wheel force-includes each vendored tree here (see the
# build config in pyproject.toml).
_STARTERS_DIR = Path(__file__).parent / "starters"
# Source-checkout layout: the same trees, where they are authored and tested.
_CHECKOUT_STARTERS_DIR = Path(__file__).resolve().parents[2] / "environments"


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


def _vendored_root(preset: EnvironmentPreset) -> Path:
    """Locate ``preset``'s bundled starter tree.

    Installs get it from the package. A source checkout has no ``starters/``
    directory — the build force-includes it from ``environments/`` — so fall
    back there to keep ``uv run hud init`` working from the repo.
    """
    for root in (_STARTERS_DIR, _CHECKOUT_STARTERS_DIR):
        candidate = root / preset.id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"bundled starter {preset.id!r} is missing from this installation")


# Present only in a source checkout where the starter's own test flow ran;
# wheel content is already filtered by the build.
_COPY_SKIP_DIRS = frozenset({".venv", "__pycache__", ".pytest_cache", ".ruff_cache"})


def _copy_vendored(preset: EnvironmentPreset, target: Path) -> None:
    """Copy the bundled starter tree into ``target``, preserving file modes."""
    root = _vendored_root(preset)
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(root)
        if _COPY_SKIP_DIRS.intersection(rel.parts):
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def materialize_preset(preset: EnvironmentPreset, target: Path) -> None:
    """Write ``preset``'s starter tree into ``target``.

    Vendored presets are copied from the tree bundled with this package, so they
    always match the installed SDK. The rest download the repo's ``main`` archive
    from ``codeload.github.com`` (not the rate-limited API) — no ``git`` required.
    The archive's top-level ``<repo>-main/`` component is stripped and any entry
    that would escape ``target`` is refused (path-traversal guard). Honors
    ``GITHUB_TOKEN`` if set.
    """
    if preset.vendored:
        _copy_vendored(preset, target)
        return

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
