"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import typer

from hud.cli.utils.build_display import display_build_summary
from hud.cli.utils.build_logs import poll_build_status, stream_build_logs
from hud.cli.utils.config import parse_env_file
from hud.cli.utils.context import create_build_context_tarball, format_size
from hud.cli.utils.environment import (
    find_dockerfile,
    get_environment_name,
    is_environment_directory,
)
from hud.cli.utils.validation import validate_environment
from hud.utils.hud_console import HUDConsole

LOGGER = logging.getLogger(__name__)


def _peek_env_keys(env_path: Path) -> list[str]:
    """Return the variable names from a .env file without loading values."""
    try:
        contents = env_path.read_text(encoding="utf-8")
        parsed = parse_env_file(contents)
        return sorted(parsed.keys())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Environment variable collection
# ---------------------------------------------------------------------------


def _handle_name_conflict(
    error: Any,
    console: HUDConsole,
) -> str | None:
    """Handle a 409 name conflict from trigger-direct. Returns registry_id to rebuild, or None."""
    try:
        detail = error.response.json().get("detail", {})
    except Exception:
        console.error("Environment name already exists on your team")
        return None

    if not isinstance(detail, dict):
        console.error(f"Environment name conflict: {detail}")
        return None

    existing_name = detail.get("existing_name", "unknown")
    existing_id = detail.get("existing_registry_id", "")

    console.warning(f"Environment '{existing_name}' already exists on your team")
    console.info(f"  Registry ID: {existing_id[:8]}...")
    console.info("")
    console.info("  (1) Link to existing environment and rebuild it")
    console.info("  (2) Cancel")

    try:
        choice = input("\n  Select [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.info("\n  Cancelled.")
        return None

    if choice == "1":
        console.info(f"  Linking to existing: {existing_id[:8]}...")
        return existing_id

    console.info("  Cancelled. Use --name to choose a different name.")
    return None


def collect_environment_variables(
    directory: Path,
    env_flags: list[str] | None,
    env_file: str | None,
    console: HUDConsole,
    *,
    skip_dotenv: bool = False,
) -> dict[str, str]:
    """Collect environment variables from various sources.

    Priority (highest to lowest):
    1. --env KEY=VALUE flags
    2. --env-file specified file
    3. .env file in directory (if exists and not skipped)

    Args:
        directory: Environment directory
        env_flags: List of KEY=VALUE strings from --env flags
        env_file: Path to env file (overrides .env)
        console: HUDConsole for output
        skip_dotenv: When True, skip auto-loading .env (--no-env or syncEnv=false)

    Returns:
        Combined environment variables dict
    """
    env_vars: dict[str, str] = {}

    if not skip_dotenv:
        env_path = Path(env_file) if env_file else directory / ".env"
        if env_path.exists():
            console.info(f"Loading environment variables from {env_path}")
            try:
                contents = env_path.read_text(encoding="utf-8")
                env_vars = parse_env_file(contents)
            except Exception as e:
                console.warning(f"Failed to parse env file: {e}")

    if env_flags:
        for flag in env_flags:
            if "=" in flag:
                key, value = flag.split("=", 1)
                env_vars[key.strip()] = value.strip()
            else:
                console.warning(f"Invalid --env format: {flag} (expected KEY=VALUE)")

    return env_vars


def deploy_environment(
    directory: str = ".",
    name: str | None = None,
    env: list[str] | None = None,
    env_file: str | None = None,
    no_env: bool = False,
    no_cache: bool = False,
    verbose: bool = False,
    registry_id: str | None = None,
    build_args: list[str] | None = None,
    build_secrets: list[str] | None = None,
) -> None:
    """Deploy a HUD environment to the platform.

    This command:
    1. Creates a tarball of your build context
    2. Uploads it to HUD's build service
    3. Triggers a remote build via CodeBuild
    4. Streams build logs in real-time
    5. Displays a summary when complete

    Args:
        directory: Environment directory containing Dockerfile
        name: Environment display name (defaults to directory name)
        env: List of KEY=VALUE environment variables
        env_file: Path to .env file (default: .env in directory)
        no_env: Skip .env file loading for this deploy
        no_cache: Disable build cache
        verbose: Show detailed output
        registry_id: Existing registry ID for rebuilds
        build_args: List of KEY=VALUE Docker build arguments
        build_secrets: List of Docker build secrets (e.g. id=GITHUB_TOKEN,env=GITHUB_TOKEN)
    """
    hud_console = HUDConsole()
    hud_console.header("HUD Environment Deploy")

    env_dir = Path(directory).resolve()

    from hud.cli.utils.api import require_api_key

    require_api_key("deploy environments")

    # Check for Dockerfile
    dockerfile = find_dockerfile(env_dir)
    if not dockerfile:
        hud_console.error("No Dockerfile.hud or Dockerfile found")
        hud_console.info(f"Directory: {env_dir}")
        hud_console.info("\nCreate a Dockerfile.hud with your environment setup.")
        hud_console.info("Run 'hud init' to create a template.")
        raise typer.Exit(1)

    hud_console.info(f"Using Dockerfile: {dockerfile.name}")

    # Pre-deploy validation - catch common issues before uploading
    hud_console.progress_message("Validating environment...")
    validation_issues = validate_environment(env_dir)

    errors = [i for i in validation_issues if i.severity == "error"]
    warnings = [i for i in validation_issues if i.severity == "warning"]

    if errors:
        hud_console.error(f"Found {len(errors)} validation error(s):")
        for issue in errors:
            file_info = f" ({issue.file})" if issue.file else ""
            hud_console.error(f"  {issue.message}{file_info}")
            if issue.hint:
                hud_console.dim_info("    Hint:", issue.hint)
        hud_console.info("")
        hud_console.info("Fix these errors before deploying.")
        raise typer.Exit(1)

    if warnings:
        hud_console.warning(f"Found {len(warnings)} warning(s):")
        for issue in warnings:
            file_info = f" ({issue.file})" if issue.file else ""
            hud_console.warning(f"  {issue.message}{file_info}")
            if issue.hint:
                hud_console.dim_info("    Hint:", issue.hint)
        hud_console.info("")

    if not validation_issues:
        hud_console.success("Validation passed")

    # Load existing config for registry_id (config.json, auto-migrates deploy.json)
    from hud.cli.utils.project_config import load_project_config

    project_config = load_project_config(env_dir)
    if not registry_id:
        registry_id = project_config.get("registryId")
        if registry_id:
            hud_console.info(f"Rebuilding existing environment: {registry_id[:8]}...")

    # Determine environment name:
    # - For rebuilds: resolve the actual name from platform (not --name flag)
    # - For new deploys: use --name flag or directory name
    if not name:
        name, _name_source = get_environment_name(env_dir, None)

    # For rebuilds, resolve actual name from platform (--name doesn't rename)
    from hud.cli.utils.api import hud_headers as _headers
    from hud.cli.utils.name_check import check_and_fix_env_name, resolve_registry_name
    from hud.settings import settings as _settings

    if registry_id:
        platform_name = resolve_registry_name(registry_id, _settings.hud_api_url, _headers())
        if platform_name:
            if name and name != platform_name:
                hud_console.warning(
                    f"--name '{name}' differs from the deployed name '{platform_name}'."
                )
            name = platform_name

    hud_console.info(f"Environment name: {name}")
    check_and_fix_env_name(env_dir, name, hud_console)

    # Resolve whether to include .env vars
    # When --env flags are explicit, they are the ONLY env vars used (no .env auto-load).
    # When no --env flags, use saved syncEnv preference or prompt.
    has_explicit_env = bool(env) or bool(env_file)
    skip_dotenv = no_env or has_explicit_env

    if not skip_dotenv:
        dotenv_path = env_dir / ".env"
        if dotenv_path.exists():
            sync_pref = project_config.get("syncEnv")

            if sync_pref is None:
                keys = _peek_env_keys(dotenv_path)
                if keys:
                    hud_console.info(f"Found .env with {len(keys)} variable(s): {', '.join(keys)}")
                    try:
                        answer = (
                            input("Include in deploy? (encrypted at rest) [Y/n]: ").strip().lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        answer = "n"
                    sync_pref = answer in ("", "y", "yes")
                    from hud.cli.utils.project_config import save_project_config as _save_cfg

                    _save_cfg({"syncEnv": sync_pref}, env_dir)
                    hud_console.dim_info("Preference saved to:", ".hud/config.json")
                else:
                    sync_pref = False

            if sync_pref:
                keys = _peek_env_keys(dotenv_path)
                hud_console.info(
                    f"Syncing {len(keys)} env var(s) from .env (saved, use --no-env to skip)"
                )
            else:
                skip_dotenv = True

    env_vars = collect_environment_variables(
        env_dir, env, env_file, hud_console, skip_dotenv=skip_dotenv
    )
    if env_vars and verbose:
        hud_console.info(f"Environment variables: {', '.join(env_vars.keys())}")

    # Parse build arguments
    build_args_dict: dict[str, str] = {}
    if build_args:
        for arg in build_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                build_args_dict[key.strip()] = value.strip()
            else:
                hud_console.warning(f"Invalid --build-arg format: {arg} (expected KEY=VALUE)")
    if build_args_dict and verbose:
        hud_console.info(f"Build arguments: {', '.join(build_args_dict.keys())}")

    build_secrets_dict: dict[str, str] = {}
    if build_secrets:
        for secret_spec in build_secrets:
            # Parse Docker secret spec: comma-separated key=value pairs
            # e.g. "id=GITHUB_TOKEN,env=GITHUB_TOKEN" or "id=mykey,src=./mykey.txt"
            parts = {}
            for part in secret_spec.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    parts[k.strip()] = v.strip()

            secret_id = parts.get("id")
            if not secret_id:
                hud_console.error(f"Invalid --secret format: {secret_spec} (missing id=)")
                raise typer.Exit(1)

            if "env" in parts:
                env_name = parts["env"]
                value = os.environ.get(env_name)
                if value is None:
                    hud_console.error(
                        f"Secret '{secret_id}': environment variable '{env_name}' is not set"
                    )
                    raise typer.Exit(1)
                build_secrets_dict[secret_id] = value
            elif "src" in parts:
                src_path = Path(parts["src"]).expanduser()
                if not src_path.is_absolute():
                    src_path = env_dir / src_path
                if not src_path.exists():
                    hud_console.error(f"Secret '{secret_id}': file not found: {src_path}")
                    raise typer.Exit(1)
                try:
                    build_secrets_dict[secret_id] = src_path.read_text(encoding="utf-8")
                except Exception as e:
                    hud_console.error(f"Secret '{secret_id}': failed to read {src_path}: {e}")
                    raise typer.Exit(1) from e
            else:
                hud_console.error(f"Invalid --secret format: {secret_spec} (need env= or src=)")
                raise typer.Exit(1)
    # Create build context tarball
    hud_console.progress_message("Creating build context tarball...")

    try:
        tarball_path, tarball_size, file_count, tarball_duration = create_build_context_tarball(
            env_dir,
            verbose=verbose,
        )
    except Exception as e:
        hud_console.error(f"Failed to create build context: {e}")
        raise typer.Exit(1) from e

    size_str = format_size(tarball_size)
    msg = f"Created tarball: {size_str} ({file_count} files) [{tarball_duration:.1f}s]"
    hud_console.success(msg)

    # Run async deployment
    try:
        result = asyncio.run(
            _deploy_async(
                tarball_path=tarball_path,
                name=name,
                env_vars=env_vars,
                build_args=build_args_dict,
                build_secrets=build_secrets_dict,
                no_cache=no_cache,
                registry_id=registry_id,
                console=hud_console,
                verbose=verbose,
            )
        )
    finally:
        # Clean up tarball
        tarball_path.unlink(missing_ok=True)

    # Save deploy link as soon as we have a registry_id, regardless of build success
    # This enables rebuilds even if the first build failed
    if result.get("registry_id"):
        _save_deploy_link(env_dir, result, hud_console, env_name=name)

    if not result.get("success"):
        raise typer.Exit(1)


async def _deploy_async(
    tarball_path: Path,
    name: str,
    env_vars: dict[str, str],
    build_args: dict[str, str],
    build_secrets: dict[str, str],
    no_cache: bool,
    registry_id: str | None,
    console: HUDConsole,
    verbose: bool = False,
) -> dict:
    """Async deployment flow."""
    from hud.cli.utils.api import hud_headers
    from hud.settings import settings

    api_url = settings.hud_api_url
    headers = hud_headers()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Get presigned upload URL
        console.progress_message("Getting upload URL...")
        step_start = time.time()

        try:
            upload_response = await client.post(
                f"{api_url.rstrip('/')}/builds/upload-url",
                headers=headers,
            )
            upload_response.raise_for_status()
            upload_data = upload_response.json()
        except httpx.HTTPStatusError as e:
            console.error(f"Failed to get upload URL: {e.response.status_code}")
            if e.response.status_code == 401:
                console.error("Invalid API key. Get a new one at https://hud.ai/settings")
            return {"success": False}
        except Exception as e:
            console.error(f"Failed to get upload URL: {e}")
            return {"success": False}

        upload_url = upload_data["upload_url"]
        build_id = upload_data["build_id"]

        console.success(f"Got upload URL [{time.time() - step_start:.1f}s]")
        console.info(f"Build ID: {build_id}")

        # Step 2: Upload tarball to S3
        console.progress_message("Uploading build context...")
        step_start = time.time()

        try:
            with open(tarball_path, "rb") as f:  # noqa: ASYNC230
                tarball_data = f.read()

            # Use a separate client for S3 (different timeout)
            async with httpx.AsyncClient(timeout=300.0) as s3_client:
                upload_result = await s3_client.put(
                    upload_url,
                    content=tarball_data,
                    headers={"Content-Type": "application/gzip"},
                )
                upload_result.raise_for_status()

            console.success(f"Upload complete [{time.time() - step_start:.1f}s]")
        except Exception as e:
            console.error(f"Failed to upload build context: {e}")
            return {"success": False}

        # Step 3: Trigger direct build
        console.progress_message("Triggering build...")
        step_start = time.time()

        try:
            trigger_payload = {
                "build_id": build_id,
                "name": name,
                "no_cache": no_cache,
            }
            if registry_id:
                trigger_payload["registry_id"] = registry_id
            if env_vars:
                trigger_payload["environment_variables"] = env_vars
            if build_args:
                trigger_payload["build_args"] = build_args
            if build_secrets:
                trigger_payload["build_secrets"] = build_secrets

            trigger_response = await client.post(
                f"{api_url.rstrip('/')}/builds/trigger-direct",
                json=trigger_payload,
                headers=headers,
            )
            trigger_response.raise_for_status()
            trigger_data = trigger_response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                conflict = _handle_name_conflict(e, console)
                if conflict:
                    trigger_payload["registry_id"] = conflict
                    try:
                        trigger_response = await client.post(
                            f"{api_url.rstrip('/')}/builds/trigger-direct",
                            json=trigger_payload,
                            headers=headers,
                        )
                        trigger_response.raise_for_status()
                        trigger_data = trigger_response.json()
                    except Exception as retry_err:
                        console.error(f"Failed to rebuild: {retry_err}")
                        return {"success": False}
                else:
                    return {"success": False}
            else:
                console.error(f"Failed to trigger build: {e.response.status_code}")
                try:
                    error_detail = e.response.json().get("detail", "")
                    if error_detail:
                        console.error(f"Error: {error_detail}")
                except Exception:  # noqa: S110
                    pass
                return {"success": False}
        except Exception as e:
            console.error(f"Failed to trigger build: {e}")
            return {"success": False}

        build_id = trigger_data["id"]
        registry_id = trigger_data["registry_id"]

        console.success(f"Build triggered [{time.time() - step_start:.1f}s]")
        console.info(f"Build ID: {build_id}")
        console.info("")

        # Step 4: Stream logs via WebSocket
        console.section_title("Build Logs")

        try:
            final_status = await stream_build_logs(
                build_id=build_id,
                console=console,
            )
        except Exception as e:
            console.warning(f"WebSocket streaming failed: {e}")
            console.info("Falling back to polling...")

            # Fall back to polling
            status_response = await poll_build_status(
                build_id=build_id,
                console=console,
            )
            final_status = status_response.get("status", "UNKNOWN")

        # Step 5: Get final status and display summary
        try:
            status_response = await client.get(
                f"{api_url.rstrip('/')}/builds/{build_id}/status",
                headers=headers,
            )
            status_response.raise_for_status()
            status_data = status_response.json()
        except Exception as e:
            console.warning(f"Failed to get final status: {e}")
            status_data = {"status": final_status}

        # Display summary — prefer backend-returned name over local name
        display_build_summary(
            status_response=status_data,
            registry_id=registry_id or "",
            console=console,
            env_name=status_data.get("registry_name") or name,
        )

        success = final_status == "SUCCEEDED"
        if success:
            console.success("Deploy complete!")
        else:
            console.error(f"Deploy failed with status: {final_status}")

        return {
            "success": success,
            "build_id": build_id,
            "registry_id": registry_id,
            "status": final_status,
            "version": status_data.get("version"),
            "lock": status_data.get("lock"),
        }


def _save_deploy_link(
    env_dir: Path,
    result: dict[str, Any],
    console: HUDConsole,
    env_name: str | None = None,
) -> None:
    """Save deploy linking info to .hud/config.json."""
    from hud.cli.utils.project_config import save_project_config

    try:
        reg_id = result.get("registry_id")
        if reg_id:
            config_data: dict[str, Any] = {"registryId": reg_id}
            if env_name:
                config_data["registryName"] = env_name
            changed = save_project_config(config_data, env_dir)
            console.success(f"Linked to environment: {reg_id[:8]}...")
            if changed:
                console.dim_info("Config saved to:", ".hud/config.json")
    except Exception as e:
        console.warning(f"Failed to save deploy link: {e}")


def discover_environments(directory: Path) -> list[Path]:
    """Find all HUD environment subdirectories within a parent directory.

    Scans immediate children for directories containing a Dockerfile
    (Dockerfile.hud or Dockerfile) and pyproject.toml.

    Returns sorted list of environment directory paths.
    """
    if not directory.is_dir():
        return []
    return [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir() and is_environment_directory(child)
    ]


def deploy_all(
    directory: str,
    env: list[str] | None = None,
    env_file: str | None = None,
    no_env: bool = False,
    no_cache: bool = False,
    verbose: bool = False,
    build_args: list[str] | None = None,
    build_secrets: list[str] | None = None,
) -> None:
    """Deploy all HUD environments found in a directory.

    Discovers subdirectories that are valid HUD environments and deploys
    each one sequentially.
    """
    hud_console = HUDConsole()
    parent = Path(directory).resolve()

    if not parent.is_dir():
        hud_console.error(f"Directory does not exist: {directory}")
        raise typer.Exit(1)

    envs = discover_environments(parent)
    if not envs:
        hud_console.error(f"No HUD environments found in {parent}")
        hud_console.info("Expected subdirectories containing Dockerfile.hud + pyproject.toml")
        raise typer.Exit(1)

    hud_console.header("Deploy All Environments")
    hud_console.info(f"Found {len(envs)} environment(s) in {parent}:")
    for env_dir in envs:
        hud_console.info(f"  {env_dir.name}/")
    hud_console.info("")

    succeeded: list[str] = []
    failed: list[str] = []

    for i, env_dir in enumerate(envs, start=1):
        hud_console.section_title(f"[{i}/{len(envs)}] Deploying {env_dir.name}")

        try:
            deploy_environment(
                directory=str(env_dir),
                name=None,
                env=env,
                env_file=env_file,
                no_env=no_env,
                no_cache=no_cache,
                verbose=verbose,
                registry_id=None,
                build_args=build_args,
                build_secrets=build_secrets,
            )
            succeeded.append(env_dir.name)
        except (typer.Exit, SystemExit):
            LOGGER.warning("Deploy failed for environment %s", env_dir.name)
            failed.append(env_dir.name)
        except Exception:
            LOGGER.exception("Unexpected error deploying %s", env_dir.name)
            failed.append(env_dir.name)

    # Summary
    hud_console.info("")
    hud_console.header("Deploy All Summary")
    if succeeded:
        hud_console.success(f"{len(succeeded)} environment(s) deployed successfully:")
        for name in succeeded:
            hud_console.info(f"  {name}")
    if failed:
        hud_console.error(f"{len(failed)} environment(s) failed:")
        for name in failed:
            hud_console.info(f"  {name}")
        raise typer.Exit(1)


def deploy_command(
    directory: str = typer.Argument(".", help="Environment directory"),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Environment display name (defaults to directory name)",
    ),
    all_envs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Deploy all HUD environments found in directory",
    ),
    env: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--env",
        "-e",
        help="Environment variable (KEY=VALUE, repeatable)",
    ),
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help="Path to .env file (default: .env in directory)",
    ),
    no_env: bool = typer.Option(
        False,
        "--no-env",
        help="Skip .env file loading for this deploy (does not change saved preference)",
    ),
    build_args: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--build-arg",
        help="Docker build argument (KEY=VALUE, repeatable)",
    ),
    secrets: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--secret",
        help="Docker build secret, e.g. --secret id=GITHUB_TOKEN,env=GITHUB_TOKEN",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable build cache",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    registry_id: str | None = typer.Option(
        None,
        "--registry-id",
        help="Existing registry ID for rebuilds (advanced)",
        hidden=True,
    ),
) -> None:
    """🚀 Deploy HUD environment to the platform.

    [not dim]Builds and deploys your environment directly from a Dockerfile,
    without requiring a GitHub repository.

    This command:
    1. Packages your Dockerfile and build context
    2. Uploads to HUD's build service
    3. Builds remotely via AWS CodeBuild
    4. Streams build logs in real-time

    Examples:
        hud deploy                     # Deploy current directory
        hud deploy environments/browser
        hud deploy . --name my-env     # Custom name
        hud deploy . -e API_KEY=xxx    # With env vars
        hud deploy ./converted --all   # Deploy all envs in directory
        hud deploy . --build-arg NODE_ENV=production  # With build args
        hud deploy . --secret id=MY_KEY,env=MY_KEY  # With build secrets (will be encrypted at rest)
        hud deploy . --secret id=MY_KEY,src=./my_key.txt  # Secret from file
        hud deploy . --no-cache        # Force rebuild
        hud deploy . --no-env          # Skip .env for this deploy[/not dim]
    """
    if all_envs:
        deploy_all(
            directory=directory,
            env=env,
            env_file=env_file,
            no_env=no_env,
            no_cache=no_cache,
            verbose=verbose,
            build_args=build_args,
            build_secrets=secrets,
        )
        return

    deploy_environment(
        directory=directory,
        name=name,
        env=env,
        env_file=env_file,
        no_env=no_env,
        no_cache=no_cache,
        verbose=verbose,
        registry_id=registry_id,
        build_args=build_args,
        build_secrets=secrets,
    )
