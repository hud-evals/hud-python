"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import typer

from hud.cli.utils.build_display import display_build_summary
from hud.cli.utils.build_logs import poll_build_status, stream_build_logs
from hud.cli.utils.config import parse_env_file
from hud.cli.utils.context import create_build_context_tarball, format_size
from hud.cli.utils.environment import find_dockerfile, get_environment_name
from hud.cli.utils.validation import validate_environment
from hud.utils.hud_console import HUDConsole


def collect_environment_variables(
    directory: Path,
    env_flags: list[str] | None,
    env_file: str | None,
    console: HUDConsole,
) -> dict[str, str]:
    """Collect environment variables from various sources.

    Priority (highest to lowest):
    1. --env KEY=VALUE flags
    2. --env-file specified file
    3. .env file in directory (if exists)

    Args:
        directory: Environment directory
        env_flags: List of KEY=VALUE strings from --env flags
        env_file: Path to env file (overrides .env)
        console: HUDConsole for output

    Returns:
        Combined environment variables dict
    """
    env_vars: dict[str, str] = {}

    # 1. Read from .env file (or specified env_file) using existing helper
    env_path = Path(env_file) if env_file else directory / ".env"
    if env_path.exists():
        console.info(f"Loading environment variables from {env_path}")
        try:
            contents = env_path.read_text(encoding="utf-8")
            env_vars = parse_env_file(contents)
        except Exception as e:
            console.warning(f"Failed to parse env file: {e}")

    # 2. Override with --env flags
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
        no_cache: Disable build cache
        verbose: Show detailed output
        registry_id: Existing registry ID for rebuilds
        build_args: List of KEY=VALUE Docker build arguments
        build_secrets: List of Docker build secrets (e.g. id=GITHUB_TOKEN,env=GITHUB_TOKEN)
    """
    hud_console = HUDConsole()
    hud_console.header("HUD Environment Deploy")

    # Import settings lazily
    from hud.settings import settings

    env_dir = Path(directory).resolve()

    # Check for API key
    if not settings.api_key:
        hud_console.error("No HUD API key found")
        hud_console.warning("A HUD API key is required to deploy environments.")
        hud_console.info("\nTo get started:")
        hud_console.info("1. Get your API key at: https://hud.ai/settings")
        hud_console.info("2. Set it via: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)

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

    # Check for existing registry_id from .hud/deploy.json (enables rebuilds)
    deploy_link_path = env_dir / ".hud" / "deploy.json"
    if deploy_link_path.exists() and not registry_id:
        try:
            import json

            with open(deploy_link_path) as f:
                deploy_link = json.load(f)
            registry_id = deploy_link.get("registryId")
            if registry_id:
                hud_console.info(f"Rebuilding existing environment: {registry_id[:8]}...")
        except Exception:  # noqa: S110
            pass

    # Determine environment name from pyproject.toml or directory
    if not name:
        name, name_source = get_environment_name(env_dir, None)
        if name_source == "config":
            hud_console.info(f"Using name from pyproject.toml: {name}")

    hud_console.info(f"Environment name: {name}")

    # Collect environment variables
    env_vars = collect_environment_variables(env_dir, env, env_file, hud_console)
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
                api_key=settings.api_key,
                api_url=settings.hud_api_url,
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
        _save_deploy_link(env_dir, result, hud_console)

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
    api_key: str,
    api_url: str,
    console: HUDConsole,
    verbose: bool = False,
) -> dict:
    """Async deployment flow.

    Args:
        tarball_path: Path to the tarball
        name: Environment name
        env_vars: Environment variables
        build_args: Docker build arguments
        build_secrets: Resolved Docker build secrets (id -> value)
        no_cache: Whether to disable cache
        registry_id: Optional existing registry ID
        api_key: HUD API key
        api_url: HUD API URL
        console: HUDConsole for output
        verbose: Verbose mode

    Returns:
        Result dict with success status and details
    """
    headers = {"X-API-Key": api_key}

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
                api_key=api_key,
                api_url=api_url,
                console=console,
            )
        except Exception as e:
            console.warning(f"WebSocket streaming failed: {e}")
            console.info("Falling back to polling...")

            # Fall back to polling
            status_response = await poll_build_status(
                build_id=build_id,
                api_key=api_key,
                api_url=api_url,
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

        # Display summary
        display_build_summary(
            status_response=status_data,
            registry_id=registry_id or "",
            console=console,
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
    result: dict,
    console: HUDConsole,
) -> None:
    """Save deploy linking info to .hud/deploy.json.

    Similar to Vercel's .vercel/project.json - stores just the IDs
    needed to link this directory to the remote environment.

    Args:
        env_dir: Environment directory
        result: Deploy result dict
        console: HUDConsole for output
    """
    import json

    hud_dir = env_dir / ".hud"
    deploy_link_path = hud_dir / "deploy.json"

    # Create .hud directory if needed
    hud_dir.mkdir(parents=True, exist_ok=True)

    # Minimal linking data (like Vercel's projectId/orgId)
    deploy_link = {
        "registryId": result.get("registry_id"),
        "version": result.get("version"),  # Last deployed version
    }

    try:
        with open(deploy_link_path, "w") as f:
            json.dump(deploy_link, f, indent=2)
        reg_id = deploy_link["registryId"]
        if reg_id:
            console.success(f"Linked to environment: {reg_id[:8]}...")
            console.dim_info("Link stored in:", ".hud/deploy.json")
    except Exception as e:
        console.warning(f"Failed to save deploy link: {e}")


def deploy_command(
    directory: str = typer.Argument(".", help="Environment directory"),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Environment display name (defaults to directory name)",
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
        hud deploy . --build-arg NODE_ENV=production  # With build args
        hud deploy . --secret id=MY_KEY,env=MY_KEY  # With build secrets (will be encrypted at rest)
        hud deploy . --secret id=MY_KEY,src=./my_key.txt  # Secret from file
        hud deploy . --no-cache        # Force rebuild[/not dim]
    """
    deploy_environment(
        directory=directory,
        name=name,
        env=env,
        env_file=env_file,
        no_cache=no_cache,
        verbose=verbose,
        registry_id=registry_id,
        build_args=build_args,
        build_secrets=secrets,
    )
