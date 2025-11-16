from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import yaml

from hud.cli.push import push_environment
from hud.cli.utils.docker import require_docker_running
from hud.cli.utils.env_check import find_environment_dir
from hud.cli.utils.registry import extract_name_and_tag
from hud.utils.hud_console import hud_console
from hud.utils.tasks import load_tasks

if TYPE_CHECKING:
    from hud.types import Task


logger = logging.getLogger(__name__)


def _is_remote_url(url: str) -> bool:
    """Match the remote url."""
    # See if a url is a remote url
    return bool(re.match(r"^(https?:\/\/)?(www\.)?[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(\/\S*)?$", url))


def _validate_tasks(tasks: list[Task]) -> bool:
    """Validate the tasks file: return True if tasks already reference a remote MCP URL.

    A task is considered remote if any "url" field anywhere inside mcp_config
    is a valid remote URL (e.g., https://mcp.hud.ai/v3/mcp).
    """

    def _has_remote_url(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "url" and isinstance(v, str) and _is_remote_url(v):
                    return True
                if _has_remote_url(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if _has_remote_url(item):
                    return True
        return False

    for task in tasks:
        cfg = task.mcp_config or {}
        if not _has_remote_url(cfg):
            return False
    return True


def _ensure_pushed(
    env_dir: Path, lock_data: dict[str, Any], check_docker: bool = True
) -> dict[str, Any]:
    """Ensure the environment is pushed to a registry; return updated lock data."""
    pushed = bool(lock_data.get("push"))
    if not pushed:
        hud_console.warning("Environment not pushed to a registry yet.")
        if not hud_console.confirm("Push to a registry now (runs 'hud push')?", default=True):
            raise typer.Exit(1)
        # Check Docker availability before attempting a push
        if check_docker:
            require_docker_running()

        # If Docker or login is not configured, the push function will fail and halt.
        push_environment(str(env_dir), yes=True)

        # Reload lock after push
        lock_path = env_dir / "hud.lock.yaml"
        with open(lock_path) as f:
            lock_data = yaml.safe_load(f) or {}

    return lock_data


def _derive_remote_image(lock_data: dict[str, Any]) -> str:
    """Derive org/name:tag from lock file for remote MCP header.

    Preference order (new lock first, then legacy):
    1) lock_data["push"]["image_with_tag"] (exact org/name:tag that was pushed)
    2) lock_data["images"]["local"] (base name with internal version)
    3) lock_data["image"] (legacy field; may contain tag or digest)
    """
    if not isinstance(lock_data, dict):  # Defensive
        raise typer.Exit(1)

    # 1) Prefer the exact image that was pushed (org/name:tag)
    push_info = lock_data.get("push") or {}
    pushed_with_tag = str(push_info.get("image_with_tag") or "").strip()
    if pushed_with_tag:
        name, tag = extract_name_and_tag(pushed_with_tag)
        return f"{name}:{tag}"

    # 2) Fall back to the local tag recorded in the new lock schema
    images = lock_data.get("images") or {}
    local_image = str(images.get("local") or "").strip()
    if local_image:
        name, tag = extract_name_and_tag(local_image)
        return f"{name}:{tag}"

    # 3) Legacy top-level image field
    legacy_image = str(lock_data.get("image") or "").strip()
    if legacy_image:
        name, tag = extract_name_and_tag(legacy_image)
        return f"{name}:{tag}"

    # If none of the above exist, we cannot derive an image
    raise typer.Exit(1)


def _extract_existing_images(tasks: list[Task]) -> set[str]:
    """Extract all Mcp-Image references from tasks."""
    images = set()

    def _extract_from_obj(obj: Any) -> None:
        if isinstance(obj, dict):
            # Check for Mcp-Image in headers
            if "headers" in obj and isinstance(obj["headers"], dict):
                mcp_image = obj["headers"].get("Mcp-Image")
                if mcp_image:
                    images.add(mcp_image)
            # Recursively check nested objects
            for v in obj.values():
                _extract_from_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_from_obj(item)

    for task in tasks:
        if task.mcp_config:
            _extract_from_obj(task.mcp_config)

    return images


def _env_var_to_header_key(var_name: str) -> str:
    """Convert ENV_VAR style to Env-Env-Var header style.

    Example: OPENAI_API_KEY -> Env-Openai-Api-Key
    """
    parts = str(var_name).split("_")
    return f"Env-{'-'.join(part.capitalize() for part in parts)}"


def _extract_api_key_vars(lock_data: dict[str, Any]) -> set[str]:
    """Extract env var names from lock file's provided section (authoritative source).

    We only use keys listed under environment.variables.provided, and exclude HUD_API_KEY
    because Authorization already carries it.
    """
    provided_keys: set[str] = set()
    if not isinstance(lock_data, dict):
        return provided_keys
    try:
        env_section = (lock_data.get("environment") or {}).get("variables") or {}
        provided = env_section.get("provided") or {}
        for name in provided:
            provided_keys.add(str(name))
    except Exception as e:
        logger.debug("Failed to parse provided env vars from lock data: %s", e)
    provided_keys.discard("HUD_API_KEY")
    return provided_keys


def _extract_dotenv_api_key_vars(env_dir: Path) -> set[str]:
    """Parse .env for API-like variables to suggest as headers.

    We intentionally include only keys that look like secrets to avoid noise:
    any key containing one of: api, key, token, secret, password (case-insensitive).
    """
    dotenv_path = env_dir / ".env"
    detected: set[str] = set()
    if not dotenv_path.exists():
        return detected
    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            name, _ = line.split("=", 1)
            name = name.strip()
            lowered = name.lower()
            if any(s in lowered for s in ("api", "key", "token", "secret", "password")):
                detected.add(name)
    except Exception:
        # Best-effort only
        return detected
    detected.discard("HUD_API_KEY")
    return detected


def _extract_env_vars_from_docker_args(args: list[str]) -> set[str]:
    """Extract environment variable names from docker run arguments.

    Parses args like: ["run", "--rm", "-i", "-e", "API_KEY=value", "-e", "TOKEN", "image:tag"]
    Returns set of env var names (not values).
    """
    env_vars: set[str] = set()
    i = 0
    while i < len(args):
        arg = args[i]

        # Check for -e or --env flags
        if arg in ("-e", "--env"):
            if i + 1 < len(args):
                env_spec = args[i + 1]
                # Could be "KEY=value" or just "KEY"
                var_name = env_spec.split("=", 1)[0].strip()
                if var_name:
                    env_vars.add(var_name)
                i += 2
                continue
        # Check for --env=KEY=value format
        elif arg.startswith("--env="):
            env_spec = arg[6:]  # Remove "--env=" prefix
            var_name = env_spec.split("=", 1)[0].strip()
            if var_name:
                env_vars.add(var_name)

        i += 1

    env_vars.discard("HUD_API_KEY")
    return env_vars


def _extract_vars_from_task_configs(raw_tasks: list[dict[str, Any]]) -> set[str]:
    """Extract environment variable names from docker run commands in task mcp_configs."""
    all_env_vars: set[str] = set()

    for task in raw_tasks:
        mcp_config = task.get("mcp_config", {})

        # Iterate through all server configs
        for server_config in mcp_config.values():
            if not isinstance(server_config, dict):
                continue

            command = server_config.get("command", "")
            args = server_config.get("args", [])

            # Only process docker run commands
            if command == "docker" and "run" in args:
                env_vars = _extract_env_vars_from_docker_args(args)
                all_env_vars.update(env_vars)

    return all_env_vars


def convert_tasks_to_remote(tasks_file: str) -> str:
    """Convert a local tasks file to remote MCP tasks and return new filename.

    Steps:
    1) Find env dir; ensure built (hud.lock.yaml), otherwise build
    2) Ensure pushed to registry, otherwise push
    3) Check for outdated images in existing task configurations
    4) Create remote_[tasks].json with mcp_config pointing to mcp.hud.ai and Mcp-Image
    5) Return the new tasks file path
    """
    tasks_path = Path(tasks_file).resolve()

    # Load validated tasks for decision-making (may resolve env vars)
    tasks: list[Task] = load_tasks(str(tasks_path))  # type: ignore[assignment]

    # Load raw tasks to preserve placeholders when writing back to disk
    raw_tasks: list[dict[str, Any]] = load_tasks(str(tasks_path), raw=True)  # type: ignore[assignment]

    # Ensure HUD_API_KEY is available: prefer process env, else load from env_dir/.env
    from hud.settings import settings

    if not settings.api_key or not settings.api_key.strip():
        hud_console.error("HUD_API_KEY is not set")
        hud_console.info("Set it in your environment or run: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Check if tasks already have remote URLs
    already_remote = _validate_tasks(tasks)

    # Extract existing images from tasks
    existing_images = _extract_existing_images(tasks)

    # Locate environment
    env_dir = find_environment_dir(tasks_path)
    if not env_dir:
        if already_remote:
            return str(tasks_path)
        hud_console.error("Could not locate an environment directory (Dockerfile + pyproject.toml)")
        hud_console.hint("Ensure you're in or near your environment folder before running 'hud rl'")
        raise typer.Exit(1)

    # For convert command, we don't need Docker running - just check for lock file
    # This avoids showing Docker-related messages during conversion
    lock_path = env_dir / "hud.lock.yaml"
    if not lock_path.exists():
        hud_console.error("No hud.lock.yaml found. The environment needs to be built first.")
        hud_console.info("Run 'hud build' in the environment directory to build it.")
        raise typer.Exit(1)

    # Load lock data directly
    try:
        with open(lock_path) as f:
            lock_data: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as e:
        hud_console.error(f"Failed to read hud.lock.yaml: {e}")
        raise typer.Exit(1) from e

    # Check if pushed - don't check Docker for convert command
    lock_data = _ensure_pushed(env_dir, lock_data, check_docker=False)

    # Derive remote image name org/name:tag
    remote_image = _derive_remote_image(lock_data)

    # Check if existing images are outdated
    needs_update = False
    should_update_image = False
    if existing_images:
        # Check if any existing image differs from the latest
        for existing_img in existing_images:
            if existing_img != remote_image:
                hud_console.warning(f"Detected outdated image reference: {existing_img}")
                hud_console.info(f"Latest pushed image: {remote_image}")
                needs_update = True
                break

        if needs_update:
            confirm_msg = "Update task configuration with the latest image?"
            if hud_console.confirm(confirm_msg, default=True):
                hud_console.info("Updating task configuration with latest image...")
                should_update_image = True
            else:
                # If user doesn't want to update, just return the original file
                if already_remote:
                    return str(tasks_path)
                # Otherwise, continue with conversion but keep old images
                remote_image = next(iter(existing_images))  # Use the first existing image

    # If tasks are already remote and up-to-date (no update needed), return original file
    if already_remote and not needs_update:
        return str(tasks_path)

    # If tasks are already remote and we just need to update the image
    if already_remote and should_update_image:
        # Update image references in-place on RAW tasks (preserve placeholders)
        def _update_image_refs_raw(obj: Any) -> Any:
            if isinstance(obj, dict):
                new_obj = {}
                for k, v in obj.items():
                    if k == "Mcp-Image" and isinstance(v, str) and v in existing_images:
                        new_obj[k] = remote_image
                    else:
                        new_obj[k] = _update_image_refs_raw(v)
                return new_obj
            elif isinstance(obj, list):
                return [_update_image_refs_raw(item) for item in obj]
            else:
                return obj

        updated_raw_tasks: list[dict[str, Any]] = []
        for t in raw_tasks:
            td = dict(t)
            if "mcp_config" in td:
                td["mcp_config"] = _update_image_refs_raw(td["mcp_config"])
            updated_raw_tasks.append(td)

        # Write updated file (preserve original format - check if it's .jsonl)
        if tasks_path.suffix == ".jsonl":
            with open(tasks_path, "w", encoding="utf-8") as f:
                for task in updated_raw_tasks:
                    json.dump(task, f, ensure_ascii=False)
                    f.write("\n")
        else:
            with open(tasks_path, "w", encoding="utf-8") as f:
                json.dump(updated_raw_tasks, f, ensure_ascii=False, indent=2)
                f.write("\n")

        hud_console.success(f"Updated {tasks_path.name} with latest image: {remote_image}")
        return str(tasks_path)

    # Extract environment variables from multiple sources:
    # 1. Lock file (authoritative for required env vars)
    provided_keys = _extract_api_key_vars(lock_data)

    # 2. Task configs (docker run -e flags)
    task_env_vars = _extract_vars_from_task_configs(raw_tasks)

    # 3. .env file (detect API-like vars)
    dotenv_keys = _extract_dotenv_api_key_vars(env_dir)

    # Combine: lock file vars + task config vars, then check for missing from .env
    all_detected = provided_keys | task_env_vars

    # If .env contains API-like vars not yet included, offer to add them
    missing = sorted(dotenv_keys - all_detected)
    if missing:
        names_preview = ", ".join(missing)
        prompt = (
            f"Detected env vars in .env that look like API keys: {names_preview}.\n"
            "Include them as remote headers (values will be ${VAR} placeholders)?"
        )
        if not hud_console.confirm(prompt, default=True):
            # User cancelled - exit without creating the file
            hud_console.info("Conversion cancelled by user")
            raise typer.Exit(0)
        all_detected.update(missing)

    # Final set of env vars to convert to headers
    provided_keys = all_detected

    extra_api_key_headers: dict[str, str] = {}
    for var_name in provided_keys:
        if str(var_name).upper() == "HUD_API_KEY":
            continue
        header_key = _env_var_to_header_key(var_name)
        extra_api_key_headers[header_key] = f"${{{var_name}}}"

    # Helper to strip extra fields from tool calls
    def _simplify_tool_call(tool: Any) -> Any:
        def _one(x: Any) -> dict[str, Any]:
            try:
                data = x.model_dump() if hasattr(x, "model_dump") else dict(x)
            except Exception:
                try:
                    data = dict(x)
                except Exception:
                    return {}
            # Keep only name and arguments
            name = data.get("name")
            arguments = data.get("arguments", {})
            return {"name": name, "arguments": arguments}

        if tool is None:
            return None
        if isinstance(tool, list):
            return [_one(x) for x in tool]
        return _one(tool)

    # Convert to list[dict]
    tasks_payload: list[dict[str, Any]] = []
    for t in tasks:
        item: dict[str, Any] = {
            "prompt": t.prompt,
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.ai/v3/mcp",
                    "headers": {
                        "Authorization": "Bearer ${HUD_API_KEY}",
                        "Mcp-Image": remote_image,
                    },
                }
            },
        }

        # Merge additional API key headers
        item["mcp_config"]["hud"]["headers"].update(extra_api_key_headers)

        # Optional fields, omit Nones
        if t.setup_tool is not None:
            item["setup_tool"] = _simplify_tool_call(t.setup_tool)
        if t.evaluate_tool is not None:
            item["evaluate_tool"] = _simplify_tool_call(t.evaluate_tool)
        if t.agent_config is not None:
            item["agent_config"] = t.agent_config
        if t.metadata:
            item["metadata"] = t.metadata
        if t.id is not None:
            item["id"] = t.id

        tasks_payload.append(item)

    remote_name = f"remote_{tasks_path.stem}.json"
    remote_path = tasks_path.parent / remote_name
    with open(remote_path, "w", encoding="utf-8") as f:
        json.dump(tasks_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    hud_console.success(f"Created remote tasks file: {remote_path.name}")

    return str(remote_path)
