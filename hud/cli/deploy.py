"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer

from hud.cli.utils.build_display import display_build_summary
from hud.cli.utils.build_logs import poll_build_status, stream_build_logs
from hud.cli.utils.config import parse_env_file, parse_key_value
from hud.cli.utils.context import create_build_context_tarball, format_size
from hud.cli.utils.registry import get_registry_environment
from hud.cli.utils.source import EnvironmentSource, normalize_environment_name
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

LOGGER = logging.getLogger(__name__)
_VALID_RUNTIMES = {"modal"}


@dataclass(frozen=True)
class _DeployPlan:
    name: str
    registry_id: str | None
    runtime: str | None
    env_vars: dict[str, str]
    build_args: dict[str, str]
    build_secrets: dict[str, str]


def _peek_env_keys(env_path: Path) -> list[str]:
    """Return the variable names from a .env file without loading values."""
    try:
        contents = env_path.read_text(encoding="utf-8")
        parsed = parse_env_file(contents)
        return sorted(parsed.keys())
    except Exception:
        return []


def _parse_key_value_flags(
    flags: list[str] | None,
    *,
    option: str,
    console: HUDConsole,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for flag in flags or []:
        parsed = parse_key_value(flag)
        if parsed is None:
            console.warning(f"Invalid {option} format: {flag} (expected KEY=VALUE)")
            continue
        values[parsed[0]] = parsed[1]
    return values


def _normalize_runtime(runtime: str | None, console: HUDConsole) -> str | None:
    if runtime is None:
        return None
    normalized = runtime.strip().lower()
    if normalized in _VALID_RUNTIMES:
        return normalized
    console.error(
        f"Invalid runtime {runtime!r}; expected one of: {', '.join(sorted(_VALID_RUNTIMES))}"
    )
    raise typer.Exit(1)


def _load_env_vars(path: Path, console: HUDConsole, *, warn_missing: bool) -> dict[str, str]:
    if not path.exists():
        if warn_missing:
            console.warning(f"Env file not found: {path}")
        return {}

    console.info(f"Loading environment variables from {path}")
    try:
        return parse_env_file(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.warning(f"Failed to parse env file: {e}")
        return {}


def collect_environment_variables(
    directory: Path,
    env_flags: list[str] | None,
    env_file: str | None,
    console: HUDConsole,
    *,
    skip_dotenv: bool = False,
) -> dict[str, str]:
    """Collect deploy environment variables from .env/--env-file plus --env overrides."""
    if env_file:
        env_vars = _load_env_vars(Path(env_file), console, warn_missing=True)
    elif not skip_dotenv:
        env_vars = _load_env_vars(directory / ".env", console, warn_missing=False)
    else:
        env_vars = {}

    env_vars.update(_parse_key_value_flags(env_flags, option="--env", console=console))
    return env_vars


def _validate_before_deploy(env_source: EnvironmentSource, console: HUDConsole) -> None:
    console.progress_message("Validating environment...")
    validation_issues = env_source.validate()

    errors = [issue for issue in validation_issues if issue.severity == "error"]
    warnings = [issue for issue in validation_issues if issue.severity == "warning"]

    if errors:
        console.error(f"Found {len(errors)} validation error(s):")
        for issue in errors:
            file_info = f" ({issue.file})" if issue.file else ""
            console.error(f"  {issue.message}{file_info}")
            if issue.hint:
                console.dim_info("    Hint:", issue.hint)
        console.info("")
        console.info("Fix these errors before deploying.")
        raise typer.Exit(1)

    if warnings:
        console.warning(f"Found {len(warnings)} warning(s):")
        for issue in warnings:
            file_info = f" ({issue.file})" if issue.file else ""
            console.warning(f"  {issue.message}{file_info}")
            if issue.hint:
                console.dim_info("    Hint:", issue.hint)
        console.info("")

    if not validation_issues:
        console.success("Validation passed")


def _resolve_declared_name(env_source: EnvironmentSource, console: HUDConsole) -> str | None:
    """The environment name declared in code, or None for legacy MCP projects.

    Prefers the Environment served by the Dockerfile entrypoint
    (``hud serve module:attr``), so a project may define auxiliary in-process
    Environments — e.g. a verification sub-agent — without making the
    deployable identity ambiguous. Otherwise a lone declared name wins, and the
    choice is only an error when nothing disambiguates between several names.
    """
    served = env_source.served_environment_name()
    if served is not None:
        return served

    references = env_source.environment_name_references()
    if not references:
        return None

    named = sorted({ref.name for ref in references if ref.name is not None})

    if len(named) > 1:
        console.error("Multiple Environment names declared in source:")
        for ref in references:
            if ref.name is not None:
                console.error(f"  {ref.file.relative_to(env_source.root)}:{ref.line}: {ref.text}")
        console.info(
            "Name the served Environment via the Dockerfile entrypoint "
            "(e.g. `hud serve env:env`), or declare exactly one name."
        )
        raise typer.Exit(1)

    if not named:
        console.error("Environment(...) is constructed without an explicit name:")
        for ref in references:
            console.error(f"  {ref.file.relative_to(env_source.root)}:{ref.line}: {ref.text}")
        console.info('Give your environment a literal name, e.g. Environment("my-env").')
        raise typer.Exit(1)

    return named[0]


def _resolve_environment_name(
    env_source: EnvironmentSource,
    registry_id: str | None,
    platform: PlatformClient,
    console: HUDConsole,
) -> str:
    """Resolve the environment name from source code.

    The name declared in ``Environment(...)`` is the environment's identity:
    the platform resolves the target registry by this name (get-or-rebuild).
    Projects without an ``Environment(...)`` call (legacy MCP environments)
    fall back to the directory name.
    """
    declared = _resolve_declared_name(env_source, console)
    name = declared if declared is not None else env_source.environment_name()

    if registry_id:
        registry_env = get_registry_environment(platform, registry_id)
        if registry_env is not None:
            if declared is not None and normalize_environment_name(name) != registry_env.name:
                console.error(
                    f"Code declares Environment('{name}') but --registry-id targets "
                    f"'{registry_env.name}'. Rename the environment in code or drop "
                    "--registry-id to deploy by name."
                )
                raise typer.Exit(1)
            if declared is None:
                name = registry_env.name

    console.info(f"Environment name: {name}")
    return name


def _skip_dotenv(
    env_source: EnvironmentSource,
    env_dir: Path,
    source_config: dict[str, Any],
    *,
    no_env: bool,
    env_file: str | None,
    console: HUDConsole,
) -> bool:
    if no_env or env_file:
        return True

    dotenv_path = env_dir / ".env"
    if not dotenv_path.exists():
        return False

    sync_pref = source_config.get("syncEnv")
    if sync_pref is None:
        keys = _peek_env_keys(dotenv_path)
        if not keys:
            return True
        console.info(f"Found .env with {len(keys)} variable(s): {', '.join(keys)}")
        sync_pref = console.confirm("Include in deploy? (encrypted at rest)")
        env_source.save_config({"syncEnv": sync_pref})
        console.dim_info("Preference saved to:", ".hud/config.json")

    if not sync_pref:
        return True

    keys = _peek_env_keys(dotenv_path)
    console.info(f"Syncing {len(keys)} env var(s) from .env (saved, use --no-env to skip)")
    return False


def _collect_build_secrets(
    secret_specs: list[str] | None,
    *,
    env_dir: Path,
    console: HUDConsole,
) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for secret_spec in secret_specs or []:
        parts: dict[str, str] = {}
        for part in secret_spec.split(","):
            key, sep, value = part.partition("=")
            if sep:
                parts[key.strip()] = value.strip()
        secret_id = parts.get("id")
        if not secret_id:
            console.error(f"Invalid --secret format: {secret_spec} (missing id=)")
            raise typer.Exit(1)

        if "env" in parts:
            env_name = parts["env"]
            value = os.environ.get(env_name)
            if value is None:
                console.error(f"Secret '{secret_id}': environment variable '{env_name}' is not set")
                raise typer.Exit(1)
            secrets[secret_id] = value
            continue

        if "src" in parts:
            src_path = Path(parts["src"]).expanduser()
            if not src_path.is_absolute():
                src_path = env_dir / src_path
            if not src_path.exists():
                console.error(f"Secret '{secret_id}': file not found: {src_path}")
                raise typer.Exit(1)
            try:
                secrets[secret_id] = src_path.read_text(encoding="utf-8")
            except OSError as e:
                console.error(f"Secret '{secret_id}': failed to read {src_path}: {e}")
                raise typer.Exit(1) from e
            continue

        console.error(f"Invalid --secret format: {secret_spec} (need env= or src=)")
        raise typer.Exit(1)
    return secrets


def _create_tarball(env_dir: Path, *, verbose: bool, console: HUDConsole) -> Path:
    console.progress_message("Creating build context tarball...")
    try:
        tarball_path, tarball_size, file_count, tarball_duration = create_build_context_tarball(
            env_dir,
            verbose=verbose,
        )
    except Exception as e:
        console.error(f"Failed to create build context: {e}")
        raise typer.Exit(1) from e

    console.success(
        f"Created tarball: {format_size(tarball_size)} ({file_count} files) "
        f"[{tarball_duration:.1f}s]"
    )
    return tarball_path


def _prepare_deploy_plan(
    env_source: EnvironmentSource,
    *,
    env_dir: Path,
    env: list[str] | None,
    env_file: str | None,
    no_env: bool,
    registry_id: str | None,
    build_args: list[str] | None,
    build_secrets: list[str] | None,
    runtime: str | None,
    verbose: bool,
    platform: PlatformClient,
    console: HUDConsole,
) -> _DeployPlan:
    source_config = env_source.load_config()
    resolved_name = _resolve_environment_name(
        env_source,
        registry_id,
        platform,
        console,
    )
    skip_dotenv = _skip_dotenv(
        env_source,
        env_dir,
        source_config,
        no_env=no_env,
        env_file=env_file,
        console=console,
    )

    env_vars = collect_environment_variables(
        env_dir,
        env,
        env_file,
        console,
        skip_dotenv=skip_dotenv,
    )
    if env and not skip_dotenv and not env_file and env_vars and (env_dir / ".env").exists():
        console.dim_info("Env merge:", ".env + --env flags (--env values take priority)")
    if env_vars and verbose:
        console.info(f"Environment variables: {', '.join(env_vars.keys())}")

    build_args_dict = _parse_key_value_flags(build_args, option="--build-arg", console=console)
    if build_args_dict and verbose:
        console.info(f"Build arguments: {', '.join(build_args_dict.keys())}")

    return _DeployPlan(
        name=resolved_name,
        registry_id=registry_id,
        runtime=_normalize_runtime(runtime, console),
        env_vars=env_vars,
        build_args=build_args_dict,
        build_secrets=_collect_build_secrets(build_secrets, env_dir=env_dir, console=console),
    )


def deploy_environment(
    directory: str = ".",
    env: list[str] | None = None,
    env_file: str | None = None,
    no_env: bool = False,
    no_cache: bool = False,
    verbose: bool = False,
    registry_id: str | None = None,
    build_args: list[str] | None = None,
    build_secrets: list[str] | None = None,
    runtime: str | None = None,
) -> None:
    """Deploy one HUD environment to the platform."""
    hud_console = HUDConsole()
    hud_console.header("HUD Environment Deploy")

    env_dir = Path(directory).resolve()
    env_source = EnvironmentSource.open(env_dir)

    from hud.cli.utils.api import require_api_key

    require_api_key("deploy environments")
    dockerfile = env_source.dockerfile
    if dockerfile is None:
        hud_console.error("No Dockerfile.hud or Dockerfile found")
        hud_console.info(f"Directory: {env_dir}")
        hud_console.info("\nCreate a Dockerfile.hud with your environment setup.")
        hud_console.info("Run 'hud init' to create a template.")
        raise typer.Exit(1)
    hud_console.info(f"Using Dockerfile: {dockerfile.name}")
    _validate_before_deploy(env_source, hud_console)

    platform = PlatformClient.from_settings()
    plan = _prepare_deploy_plan(
        env_source,
        env_dir=env_dir,
        env=env,
        env_file=env_file,
        no_env=no_env,
        registry_id=registry_id,
        build_args=build_args,
        build_secrets=build_secrets,
        runtime=runtime,
        verbose=verbose,
        platform=platform,
        console=hud_console,
    )
    tarball_path = _create_tarball(env_dir, verbose=verbose, console=hud_console)
    try:
        result = asyncio.run(
            _deploy_async(
                tarball_path=tarball_path,
                no_cache=no_cache,
                plan=plan,
                platform=platform,
                console=hud_console,
                env_dir=env_dir,
            )
        )
    finally:
        tarball_path.unlink(missing_ok=True)

    if not result.success:
        raise typer.Exit(1)


@dataclass(frozen=True)
class _DeployResult:
    success: bool
    build_id: str | None = None
    registry_id: str | None = None
    status: str = ""


@dataclass(frozen=True)
class _BuildUpload:
    upload_url: str
    build_id: str


async def _create_build_upload(platform: PlatformClient) -> _BuildUpload:
    data = await platform.apost("/builds/upload-url")
    return _BuildUpload(upload_url=data["upload_url"], build_id=data["build_id"])


async def _upload_build_context(upload_url: str, tarball_path: Path) -> None:
    """PUT the tarball to the presigned S3 URL (not a platform API call)."""
    content = await asyncio.to_thread(tarball_path.read_bytes)
    async with httpx.AsyncClient(timeout=300.0) as s3_client:
        response = await s3_client.put(
            upload_url,
            content=content,
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()


async def _trigger_build(
    platform: PlatformClient,
    *,
    build_id: str,
    plan: _DeployPlan,
    no_cache: bool,
    console: HUDConsole,
) -> dict[str, Any] | None:
    """Trigger the direct build. The platform resolves the registry by name
    (get-or-rebuild), so an existing environment with this name is rebuilt."""
    payload: dict[str, Any] = {
        "source": "direct",
        "build_id": build_id,
        "name": plan.name,
        "no_cache": no_cache,
    }
    if plan.registry_id:
        payload["registry_id"] = plan.registry_id
    if plan.runtime:
        payload["runtime_provider"] = plan.runtime
    if plan.env_vars:
        payload["environment_variables"] = plan.env_vars
    if plan.build_args:
        payload["build_args"] = plan.build_args
    if plan.build_secrets:
        payload["build_secrets"] = plan.build_secrets

    try:
        return await platform.apost("/builds/trigger", json=payload)
    except HudRequestError as e:
        console.error(f"Failed to trigger build: {e.status_code or e}")
        detail = (e.response_json or {}).get("detail", "")
        if detail:
            console.error(f"Error: {detail}")
        return None
    except Exception as e:
        console.error(f"Failed to trigger build: {e}")
        return None


async def _deploy_async(
    tarball_path: Path,
    no_cache: bool,
    plan: _DeployPlan,
    platform: PlatformClient,
    console: HUDConsole,
    env_dir: Path | None = None,
) -> _DeployResult:
    """Async deployment flow: upload context, trigger build, stream logs."""
    console.progress_message("Getting upload URL...")
    step_start = time.time()

    try:
        upload = await _create_build_upload(platform)
    except HudRequestError as e:
        console.error(f"Failed to get upload URL: {e.status_code or e}")
        if e.status_code == 401:
            from hud.settings import settings

            console.error(f"Invalid API key. Get a new one at {settings.hud_web_url}/settings")
        return _DeployResult(success=False)
    except Exception as e:
        console.error(f"Failed to get upload URL: {e}")
        return _DeployResult(success=False)

    console.success(f"Got upload URL [{time.time() - step_start:.1f}s]")
    console.info(f"Build ID: {upload.build_id}")

    console.progress_message("Uploading build context...")
    step_start = time.time()

    try:
        await _upload_build_context(upload.upload_url, tarball_path)
        console.success(f"Upload complete [{time.time() - step_start:.1f}s]")
    except Exception as e:
        console.error(f"Failed to upload build context: {e}")
        return _DeployResult(success=False)

    console.progress_message("Triggering build...")
    step_start = time.time()

    trigger_data = await _trigger_build(
        platform,
        build_id=upload.build_id,
        plan=plan,
        no_cache=no_cache,
        console=console,
    )
    if trigger_data is None:
        return _DeployResult(success=False)

    build_id = trigger_data["id"]
    registry_id = trigger_data["registry_id"]

    # Save immediately after trigger so rebuilds work even if streaming crashes.
    if env_dir and registry_id:
        _save_deploy_link(env_dir, registry_id, console, env_name=plan.name)

    console.success(f"Build triggered [{time.time() - step_start:.1f}s]")
    console.info(f"Build ID: {build_id}")
    console.info("")

    console.section_title("Build Logs")
    try:
        final_status = await stream_build_logs(platform, build_id, console=console)
    except Exception as e:
        console.warning(f"WebSocket streaming failed: {e}")
        console.info("Falling back to polling...")
        status_response = await poll_build_status(platform, build_id, console=console)
        final_status = status_response.get("status", "UNKNOWN")

    try:
        status_data = await platform.aget(f"/builds/{build_id}/status")
    except Exception as e:
        console.warning(f"Failed to get final status: {e}")
        status_data = {"status": final_status}

    # Display summary; prefer backend-returned name over local name.
    display_build_summary(
        status_response=status_data,
        registry_id=registry_id or "",
        console=console,
        env_name=status_data.get("registry_name") or plan.name,
    )

    success = final_status == "SUCCEEDED"
    if success:
        console.success("Deploy complete!")
    else:
        console.error(f"Deploy failed with status: {final_status}")

    return _DeployResult(
        success=success,
        build_id=build_id,
        registry_id=registry_id,
        status=final_status,
    )


def _save_deploy_link(
    env_dir: Path,
    registry_id: str,
    console: HUDConsole,
    env_name: str | None = None,
) -> None:
    """Save deploy linking info to .hud/config.json."""
    try:
        config_data: dict[str, Any] = {"registryId": registry_id}
        if env_name:
            config_data["registryName"] = env_name
        changed = EnvironmentSource.open(env_dir).save_config(config_data)
        console.success(f"Linked to environment: {registry_id[:8]}...")
        if changed:
            console.dim_info("Config saved to:", ".hud/config.json")
    except Exception as e:
        console.warning(f"Failed to save deploy link: {e}")


def discover_environments(directory: Path) -> list[Path]:
    """Find immediate child directories that contain a HUD environment."""
    if not directory.is_dir():
        return []
    return [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir() and EnvironmentSource.open(child).is_environment
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
    runtime: str | None = None,
) -> None:
    """Deploy each HUD environment under a parent directory."""
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
                env=env,
                env_file=env_file,
                no_env=no_env,
                no_cache=no_cache,
                verbose=verbose,
                registry_id=None,
                build_args=build_args,
                build_secrets=build_secrets,
                runtime=runtime,
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
    directory: str = typer.Argument(".", help="Environment directory or env.py file"),
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
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Persist Modal as the hosted runtime for this registry",
    ),
) -> None:
    """Deploy HUD environment to the platform.

    Accepts a directory or an env.py file — if a file is given, its parent
    directory is used. The environment name comes from the ``Environment(...)``
    declaration in code. Builds from the local Dockerfile and streams remote
    build logs.
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
            runtime=runtime,
        )
        return

    deploy_environment(
        directory=directory,
        env=env,
        env_file=env_file,
        no_env=no_env,
        no_cache=no_cache,
        verbose=verbose,
        registry_id=registry_id,
        build_args=build_args,
        build_secrets=secrets,
        runtime=runtime,
    )
