"""Shared lock file helpers: loading, path resolution, image extraction."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .analysis import BuildAnalysis

import yaml

from hud.cli.utils.environment import find_dockerfile
from hud.cli.utils.source_hash import compute_source_hash, list_source_files
from hud.version import __version__ as hud_version

LOCK_FILENAME = "hud.lock.yaml"


def load_lock(path: Path) -> dict[str, Any]:
    """Load and parse a hud.lock.yaml file. Raises on missing/invalid."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_lock(directory: Path) -> Path | None:
    """Find hud.lock.yaml in *directory* or its parent. Returns None if not found."""
    for candidate in [directory, directory.parent]:
        lock = candidate / LOCK_FILENAME
        if lock.exists():
            return lock
    return None


def get_local_image(lock_data: dict[str, Any]) -> str:
    """Extract the local image reference from lock data.

    Checks ``images.local`` (new format) then ``image`` (legacy).
    Returns empty string if neither exists.
    """
    return lock_data.get("images", {}).get("local") or lock_data.get("image", "")


def dump_lock_data(lock_data: dict[str, Any], *, sort_keys: bool = False) -> str:
    """Serialize lock data to YAML with stable formatting."""
    return yaml.dump(lock_data, default_flow_style=False, sort_keys=sort_keys)


def build_lock_data(
    *,
    source_dir: Path | None,
    analysis: BuildAnalysis | dict[str, Any],
    version: str,
    image_name: str,
    full_image_ref: str | None = None,
    pushed_image_ref: str | None = None,
    env_vars: dict[str, str] | None = None,
    additional_required_env_vars: set[str] | list[str] | None = None,
    hud_version_value: str | None = None,
    platform: str = "linux/amd64",
    build_id: str | None = None,
    build_method: str | None = None,
    directory_name: str | None = None,
    local_image_ref: str | None = None,
) -> dict[str, Any]:
    """Build a `hud.lock.yaml`-compatible dict from shared analysis data."""
    from hud.cli.build import extract_env_vars_from_dockerfile, parse_base_image

    resolved_source_dir = source_dir.resolve() if source_dir is not None else None
    dockerfile_path = (
        find_dockerfile(resolved_source_dir) if resolved_source_dir is not None else None
    )
    required_env, optional_env = (
        extract_env_vars_from_dockerfile(dockerfile_path)
        if dockerfile_path is not None
        else ([], [])
    )
    resolved_directory_name = directory_name or (
        resolved_source_dir.name
        if resolved_source_dir is not None
        else image_name.rsplit("/", 1)[-1].split(":", 1)[0]
    )
    resolved_local_image_ref = local_image_ref or f"{image_name}:{version}"

    lock_content: dict[str, Any] = {
        "version": "1.3",
        "images": {
            "local": resolved_local_image_ref,
            "full": full_image_ref,
            "pushed": pushed_image_ref,
        },
        "build": {
            "generatedAt": datetime.now(UTC).isoformat() + "Z",
            "hudVersion": hud_version_value or hud_version,
            "directory": resolved_directory_name,
            "version": version,
            "platform": platform,
        },
        "environment": {
            "initializeMs": int(analysis.get("initializeMs", 0) or 0),
            "toolCount": int(analysis.get("toolCount", 0) or 0),
            "internalToolCount": int(analysis.get("internalToolCount", 0) or 0),
        },
    }
    if build_id is not None:
        lock_content["build"]["buildId"] = build_id
    if build_method is not None:
        lock_content["build"]["buildMethod"] = build_method

    if dockerfile_path is not None:
        base_image = parse_base_image(dockerfile_path)
        if base_image:
            lock_content["build"]["baseImage"] = base_image

    if resolved_source_dir is not None:
        with contextlib.suppress(Exception):
            lock_content["build"]["sourceHash"] = compute_source_hash(resolved_source_dir)
        with contextlib.suppress(Exception):
            lock_content["build"]["sourceFiles"] = [
                str(path.resolve().relative_to(resolved_source_dir)).replace("\\", "/")
                for path in list_source_files(resolved_source_dir)
            ]

    required_from_extra = set(additional_required_env_vars or [])
    provided_env_vars = set((env_vars or {}).keys())
    all_required = (set(required_env) | required_from_extra | provided_env_vars) - set(optional_env)
    if all_required or optional_env:
        variables: dict[str, Any] = {
            "_note": (
                "You can edit this section to add or modify environment variables. "
                "Provided variables will be used when running the environment."
            )
        }
        if all_required:
            variables["required"] = sorted(all_required)
        if optional_env:
            variables["optional"] = optional_env
        lock_content["environment"]["variables"] = variables

    tools = analysis.get("tools") or []
    if tools:
        tools_serialized: list[dict[str, Any]] = []
        for tool in tools:
            entry: dict[str, Any] = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            if tool.get("internalTools"):
                entry["internalTools"] = tool["internalTools"]
            tools_serialized.append(entry)
        lock_content["tools"] = tools_serialized

    hub_tools = analysis.get("hubTools")
    if hub_tools:
        lock_content["hubTools"] = hub_tools
    prompts = analysis.get("prompts")
    if prompts:
        lock_content["prompts"] = prompts
    resources = analysis.get("resources")
    if resources:
        lock_content["resources"] = resources
    if "scenarios" in analysis:
        lock_content["scenarios"] = analysis.get("scenarios") or []

    return lock_content
