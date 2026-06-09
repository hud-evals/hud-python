"""The ``hud.lock.yaml`` build-lock format: read, write, fingerprint, compose."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from hud.environment.source import EnvironmentSource


def read_lock(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as file:
        return yaml.safe_load(file) or {}


def dump_lock(lock_data: dict[str, Any], *, sort_keys: bool = False) -> str:
    import yaml

    return yaml.dump(lock_data, default_flow_style=False, sort_keys=sort_keys)


def write_lock(path: Path, lock_data: dict[str, Any]) -> Path:
    path.write_text(dump_lock(lock_data), encoding="utf-8")
    return path


def lock_fingerprint(lock_data: dict[str, Any]) -> tuple[str, int]:
    content = dump_lock(lock_data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest(), len(content)


def local_image(lock_data: dict[str, Any]) -> str:
    images = lock_data.get("images")
    if isinstance(images, dict):
        local = images.get("local")
        if isinstance(local, str):
            return local
    image = lock_data.get("image")
    return image if isinstance(image, str) else ""


def build_lock_data(
    source: EnvironmentSource,
    *,
    analysis: dict[str, Any],
    version: str,
    local_image_ref: str,
    pushed_image_ref: str | None = None,
    env_vars: dict[str, str] | None = None,
    extra_required_env: Iterable[str] = (),
    platform: str = "linux/amd64",
) -> dict[str, Any]:
    """Compose lock-file content for one build of *source*.

    ``images.full`` (the digest-qualified ref) is left ``None``; the build flow
    fills it in after the image digest is known.
    """
    from hud.version import __version__ as hud_version

    lock_content: dict[str, Any] = {
        "version": "2.0",
        "images": {
            "local": local_image_ref,
            "full": None,
            "pushed": pushed_image_ref,
        },
        "build": {
            "generatedAt": datetime.now(UTC).isoformat() + "Z",
            "hudVersion": hud_version,
            "directory": source.root.name,
            "version": version,
            "platform": platform,
            "sourceHash": source.source_hash(),
            "sourceFiles": source.source_file_refs(),
        },
        "environment": {},
    }

    base_image = source.base_image()
    if base_image:
        lock_content["build"]["baseImage"] = base_image

    all_required = set(source.dockerfile_env_vars())
    all_required.update(extra_required_env)
    all_required.update((env_vars or {}).keys())
    if all_required:
        lock_content["environment"]["variables"] = {
            "_note": (
                "You can edit this section to add or modify environment variables. "
                "Provided variables will be used when running the environment."
            ),
            "required": sorted(all_required),
        }

    capabilities = analysis.get("capabilities") or []
    if capabilities:
        lock_content["capabilities"] = capabilities
    tasks = analysis.get("tasks") or []
    if tasks:
        lock_content["tasks"] = tasks

    return lock_content


__all__ = [
    "build_lock_data",
    "dump_lock",
    "local_image",
    "lock_fingerprint",
    "read_lock",
    "write_lock",
]
