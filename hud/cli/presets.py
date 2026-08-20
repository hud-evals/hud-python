"""Example environments used by ``hud init``."""

from __future__ import annotations

import io
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx
from packaging.version import Version

from hud.version import __version__


@dataclass(frozen=True, slots=True)
class EnvironmentPreset:
    name: str
    description: str


ENVIRONMENT_PRESETS: dict[str, EnvironmentPreset] = {
    "coding": EnvironmentPreset(
        "Coding",
        "A repository workspace with a SWE-bench task and hidden-test grading.",
    ),
    "cua": EnvironmentPreset(
        "Computer Use",
        "A virtual Linux desktop with deterministic and model-judged grading.",
    ),
    "blank": EnvironmentPreset(
        "Blank",
        "A minimal letter-counting task for building an environment from scratch.",
    ),
}
DEFAULT_PRESET_ID = "coding"


def materialize_preset(
    preset_id: str,
    target: Path,
) -> None:
    """Copy an example environment from this checkout or the installed SDK's release tag."""
    repository = Path(__file__).resolve().parents[2]
    local_source = repository / "environments" / preset_id
    if local_source.is_dir():
        shutil.copytree(
            local_source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
                "*.pyo",
            ),
        )
        return

    parsed_version = Version(__version__)
    if parsed_version.is_devrelease:
        raise ValueError(
            f"HUD SDK development version {__version__!r} has no matching example archive; "
            "run hud init from a source checkout"
        )

    headers = {}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    ref = f"v{parsed_version.public}"
    url = f"https://codeload.github.com/hud-evals/hud-python/tar.gz/refs/tags/{ref}"
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    source_parts = ("environments", preset_id)
    found = False

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            archive_parts = PurePosixPath(member.name).parts[1:]
            if archive_parts[: len(source_parts)] != source_parts:
                continue
            found = True
            relative_parts = archive_parts[len(source_parts) :]
            if not relative_parts:
                continue

            destination = (target_root / Path(*relative_parts)).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f"unsafe path in SDK archive: {member.name!r}") from exc

            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_file = archive.extractfile(member)
                if source_file is not None:
                    destination.write_bytes(source_file.read())
                    if member.mode & 0o111:
                        destination.chmod(destination.stat().st_mode | (member.mode & 0o111))

    if not found:
        raise ValueError(
            f"example environment {preset_id!r} was not found at environments/{preset_id} "
            f"in HUD SDK tag {ref}"
        )
