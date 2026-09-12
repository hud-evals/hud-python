"""Deploy HUD environments to the platform via direct build."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import typer
from pydantic import ValidationError

from hud.cli.utils.api import missing_api_key_error
from hud.cli.utils.build_display import display_build_summary
from hud.cli.utils.build_logs import poll_build_status, stream_build_logs
from hud.cli.utils.config import (
    AuthScope,
    DirectoryLink,
    DirectoryState,
    parse_env_file,
    parse_key_value,
)
from hud.cli.utils.context import create_build_context_tarball, format_size
from hud.cli.utils.output import (
    CliError,
    emit_json,
    is_interactive,
    json_option,
    map_exception,
    wants_json,
)
from hud.cli.utils.project import (
    PROJECT_OPTION_HELP,
    Placement,
    require_writable_placement,
    resolve_placement,
)
from hud.cli.utils.registry import get_registry_environment
from hud.cli.utils.source import EnvironmentSource
from hud.eval.runtime import ComposeProject, RuntimeConfig
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.utils.platform import PlatformClient

_VALID_RUNTIMES = {"hud", "modal"}
_COMPOSE_RECIPE_NAMES = (
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
)


@dataclass(frozen=True)
class _DeployPlan:
    name: str
    registry_id: str | None
    placement: Placement
    runtime: str | None
    runtime_config: RuntimeConfig | None
    env_vars: dict[str, str]
    build_args: dict[str, str]
    build_secrets: dict[str, str]
    state: DirectoryState
    dotenv_pending: bool = False
    dotenv_consent: bool | None = None
    save_link: bool = True


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
    raise ValueError(
        f"Invalid runtime {runtime!r}; expected one of: {', '.join(sorted(_VALID_RUNTIMES))}"
    )


def _compose_recipe(context: Path) -> Path | None:
    for name in _COMPOSE_RECIPE_NAMES:
        candidate = context / name
        if candidate.is_file():
            return candidate
    return next(
        (
            candidate
            for candidate in sorted(context.glob("docker-compose.*"))
            if ".override." not in candidate.name and candidate.is_file()
        ),
        None,
    )


def _load_runtime_config(path: str | None, console: HUDConsole) -> RuntimeConfig | None:
    if path is None:
        return None
    config_path = Path(path).expanduser()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_config = cast("dict[str, Any]", raw)
            compose = raw_config.get("compose")
            if isinstance(compose, dict):
                compose_config = cast("dict[str, Any]", compose)
                for field in ("document", "root"):
                    value = compose_config.get(field)
                    if not isinstance(value, str):
                        continue
                    candidate = Path(value).expanduser()
                    compose_config[field] = str(
                        candidate
                        if candidate.is_absolute()
                        else (config_path.parent / candidate).resolve()
                    )
        config = RuntimeConfig.model_validate(raw)
    except FileNotFoundError:
        raise ValueError(f"Runtime config file not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid runtime config JSON in {config_path}: {exc.msg}") from exc
    except ValidationError as exc:
        raise ValueError(f"Invalid runtime config in {config_path}: {exc}") from exc
    return config


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
        raise ValueError(
            "Environment validation failed: "
            + "; ".join(f"{issue.message} ({issue.file})" for issue in errors)
        )

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


def _resolve_declared_name(env_source: EnvironmentSource, console: HUDConsole) -> str:
    """Resolve the environment name declared in code.

    Prefers the Environment served by the Dockerfile entrypoint
    (``hud serve module:attr``), so a project may define auxiliary in-process
    Environments — e.g. a verification sub-agent — without making the
    deployable identity ambiguous. Otherwise a lone declared name wins, and the
    choice is only an error when nothing disambiguates between several names.
    """
    served = env_source.served_environment_name()
    if served is not None:
        return served

    names = {ref.name for ref in env_source.environment_name_references() if ref.name is not None}
    if len(names) != 1:
        raise ValueError(
            "Declare exactly one literal Environment name "
            "or select the served module in the Dockerfile."
        )
    return next(iter(names))


def _resolve_environment_name(
    env_source: EnvironmentSource,
    registry_id: str | None,
    platform: PlatformClient,
    console: HUDConsole,
) -> str:
    """Resolve the environment name from source code.

    The name declared in ``Environment(...)`` is the environment's identity:
    the platform resolves the target registry by this name (get-or-rebuild).
    Projects must declare an ``Environment(...)`` in source.
    """
    name = _resolve_declared_name(env_source, console)

    if registry_id:
        registry_env = get_registry_environment(platform, registry_id)
        if normalize_environment_name(name) != registry_env.name:
            raise ValueError(
                f"Code declares Environment('{name}') but --registry-id targets "
                f"'{registry_env.name}'. Rename the environment in code or drop "
                "--registry-id to deploy by name."
            )
    console.info(f"Environment name: {name}")
    return name


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
            raise ValueError(f"Invalid --secret format: {secret_spec} (missing id=)")

        if "env" in parts:
            env_name = parts["env"]
            value = os.environ.get(env_name)
            if value is None:
                raise ValueError(
                    f"Secret '{secret_id}': environment variable '{env_name}' is not set"
                )
            secrets[secret_id] = value
            continue

        if "src" in parts:
            src_path = Path(parts["src"]).expanduser()
            if not src_path.is_absolute():
                src_path = env_dir / src_path
            if not src_path.exists():
                raise ValueError(f"Secret '{secret_id}': file not found: {src_path}")
            try:
                secrets[secret_id] = src_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ValueError(f"Secret '{secret_id}': failed to read {src_path}: {e}") from e
            continue

        raise ValueError(f"Invalid --secret format: {secret_spec} (need env= or src=)")
    return secrets


def _create_tarball(env_dir: Path, *, verbose: bool, console: HUDConsole) -> Path:
    console.progress_message("Creating build context tarball...")
    try:
        tarball_path, tarball_size, file_count, tarball_duration = create_build_context_tarball(
            env_dir,
            verbose=verbose,
        )
    except Exception as e:
        raise ValueError(f"Failed to create build context: {e}") from e

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
    project: str | None,
    build_args: list[str] | None,
    build_secrets: list[str] | None,
    runtime: str | None,
    runtime_config: str | None,
    verbose: bool,
    platform: PlatformClient,
    console: HUDConsole,
) -> _DeployPlan:
    state = DirectoryState(AuthScope.resolve(platform), env_dir)
    link = state.load()
    linked = None
    if link.registry_id and registry_id is None:
        linked = get_registry_environment(platform, str(link.registry_id))
    resolved_name = _resolve_environment_name(
        env_source,
        registry_id,
        platform,
        console,
    )
    placement = resolve_placement(platform, link, flag=project)
    require_writable_placement(placement)
    consent = None
    if (
        linked is not None
        and linked.name == normalize_environment_name(resolved_name)
        and (placement.project_id is None or placement.project_id == linked.project_id)
    ):
        consent = link.sync_env.get(UUID(linked.id))
    dotenv_pending = (
        not no_env and not env_file and (env_dir / ".env").is_file() and consent is None
    )
    skip_dotenv = no_env or bool(env_file) or consent is not True

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
    normalized_runtime = _normalize_runtime(runtime, console)
    loaded_runtime_config = _load_runtime_config(runtime_config, console)
    recipe = _compose_recipe(env_dir)
    if recipe is not None:
        if loaded_runtime_config is not None and (
            loaded_runtime_config.image is not None or loaded_runtime_config.compose is not None
        ):
            raise ValueError("--runtime-config cannot set image or Compose for a Compose context")
        loaded_runtime_config = RuntimeConfig.model_validate(
            {
                **(
                    loaded_runtime_config.model_dump(exclude_unset=True)
                    if loaded_runtime_config is not None
                    else {}
                ),
                "compose": ComposeProject(document=recipe, root=env_dir),
            }
        )

    return _DeployPlan(
        state=state,
        dotenv_pending=dotenv_pending,
        save_link=registry_id is None and project is None,
        name=resolved_name,
        registry_id=registry_id,
        placement=placement,
        runtime=normalized_runtime,
        runtime_config=loaded_runtime_config,
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
    project: str | None = None,
    build_args: list[str] | None = None,
    build_secrets: list[str] | None = None,
    runtime: str | None = None,
    runtime_config: str | None = None,
    *,
    dry_run: bool = False,
) -> _DeployResult:
    """Prepare and execute one deployment, returning its complete result."""
    console = HUDConsole()
    env_source = EnvironmentSource.open(directory)
    env_dir = env_source.root
    from hud.settings import settings

    if not settings.api_key:
        raise missing_api_key_error("deploy environments")
    if _compose_recipe(env_dir) is None and env_source.dockerfile is None:
        raise CliError(
            "failure",
            "No compose.yaml, compose.yml, docker-compose.*, or Dockerfile found",
            input={"directory": str(env_dir)},
            suggestion="Run 'hud init' to create a template.",
        )
    _validate_before_deploy(env_source, console)
    platform = PlatformClient.from_settings()
    plan = _prepare_deploy_plan(
        env_source,
        env_dir=env_dir,
        env=env,
        env_file=env_file,
        no_env=no_env,
        registry_id=registry_id,
        project=project,
        build_args=build_args,
        build_secrets=build_secrets,
        runtime=runtime,
        runtime_config=runtime_config,
        verbose=verbose,
        platform=platform,
        console=console,
    )
    if dry_run:
        return _DeployResult(
            success=True,
            name=plan.name,
            registry_id=plan.registry_id,
            dry_run=True,
            runtime=plan.runtime,
            env_var_keys=sorted(plan.env_vars),
            build_arg_keys=sorted(plan.build_args),
            dotenv_pending=plan.dotenv_pending,
        )
    if plan.dotenv_pending:
        if not is_interactive():
            raise CliError(
                "confirmation_required",
                "Choose whether to upload .env before deploying.",
                suggestion="Pass --env-file .env to include it, or --no-env to skip it.",
            )
        consent = console.confirm("Include .env in deploy? (encrypted at rest)", default=False)
        plan = replace(
            plan,
            dotenv_consent=consent,
            env_vars=collect_environment_variables(
                env_dir,
                env,
                env_file,
                console,
                skip_dotenv=not consent,
            ),
        )
    tarball = _create_tarball(env_dir, verbose=verbose, console=console)
    try:
        return asyncio.run(
            _deploy_async(
                tarball_path=tarball,
                no_cache=no_cache,
                plan=plan,
                platform=platform,
                console=console,
                env_dir=env_dir,
            )
        )
    finally:
        tarball.unlink(missing_ok=True)


@dataclass(frozen=True)
class _DeployResult:
    success: bool
    action: str = "deploy"
    build_id: str | None = None
    registry_id: str | None = None
    status: str = ""
    name: str = ""
    dry_run: bool = False
    runtime: str | None = None
    env_var_keys: list[str] = field(default_factory=list)
    build_arg_keys: list[str] = field(default_factory=list)
    dotenv_pending: bool = False
    details: dict[str, Any] = field(default_factory=dict)


async def _upload_context(upload_url: str, tarball: Path) -> None:
    content = await asyncio.to_thread(tarball.read_bytes)
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.put(
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
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "source": "direct",
        "build_id": build_id,
        "name": plan.name,
        "no_cache": no_cache,
    }
    payload.update(
        {
            key: value
            for key, value in (
                ("registry_id", plan.registry_id),
                ("project_id", plan.placement.project_id),
                ("runtime_provider", plan.runtime),
                (
                    "runtime_config",
                    plan.runtime_config.model_dump(mode="json", exclude_unset=True)
                    if plan.runtime_config
                    else None,
                ),
                ("environment_variables", plan.env_vars),
                ("build_args", plan.build_args),
                ("build_secrets", plan.build_secrets),
            )
            if value
        }
    )
    data = await platform.apost("/builds/trigger", json=payload)
    return data["id"], data["registry_id"]


async def _deploy_async(
    tarball_path: Path,
    no_cache: bool,
    plan: _DeployPlan,
    platform: PlatformClient,
    console: HUDConsole,
    env_dir: Path | None = None,
) -> _DeployResult:
    """Narrate the library-owned exchange, stream logs, and save the link."""
    console.progress_message("Getting upload URL...")
    step_start = time.time()

    upload = await platform.apost("/builds/upload-url")
    upload_url, reserved_id = upload["upload_url"], upload["build_id"]

    console.success(f"Got upload URL [{time.time() - step_start:.1f}s]")
    console.info(f"Build ID: {reserved_id}")

    console.progress_message("Uploading build context...")
    step_start = time.time()

    await _upload_context(upload_url, tarball_path)
    console.success(f"Upload complete [{time.time() - step_start:.1f}s]")
    console.progress_message("Triggering build...")
    step_start = time.time()
    build_id, registry_id = await _trigger_build(
        platform,
        build_id=reserved_id,
        plan=plan,
        no_cache=no_cache,
    )
    if registry_id and plan.save_link:
        changes = DirectoryLink(registry_id=UUID(registry_id))
        if plan.dotenv_consent is not None:
            changes.sync_env = {UUID(registry_id): plan.dotenv_consent}
        plan.state.update(changes)

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

    return _DeployResult(
        success=final_status == "SUCCEEDED",
        details=status_data,
        name=plan.name,
        build_id=build_id,
        registry_id=registry_id,
        status=final_status,
    )


def discover_environments(directory: Path) -> list[Path]:
    """Find immediate child directories that contain a HUD environment."""
    if not directory.is_dir():
        return []
    return [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir()
        and (EnvironmentSource.open(child).is_environment or _compose_recipe(child) is not None)
    ]


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
    project: str | None = typer.Option(
        None,
        "--project",
        help=PROJECT_OPTION_HELP,
    ),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Persist a registry default runtime for tasks that do not specify one: hud or modal",
    ),
    runtime_config: str | None = typer.Option(
        None,
        "--runtime-config",
        help="Path to a JSON RuntimeConfig for hosted runs",
    ),
    json_output: bool = json_option(),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the deploy plan without uploading."
    ),
) -> None:
    """Deploy HUD environment to the platform.

    Accepts a directory or an env.py file — if a file is given, its parent
    directory is used. The environment name comes from the ``Environment(...)``
    declaration in code. Builds from the local Dockerfile and streams remote
    build logs.

    [not dim]Examples:
        hud deploy
        hud deploy --dry-run --json
        hud deploy --all --json[/not dim]
    """
    directories = (
        discover_environments(Path(directory).resolve()) if all_envs else [Path(directory)]
    )
    if not directories:
        raise CliError("not_found", f"No HUD environments found in {directory}")
    succeeded: list[str] = []
    failed: list[str] = []
    entries: list[dict[str, Any]] = []
    for target in directories:
        try:
            result = deploy_environment(
                directory=str(target),
                env=env,
                env_file=env_file,
                no_env=no_env,
                no_cache=no_cache,
                verbose=verbose,
                registry_id=registry_id,
                project=project,
                build_args=build_args,
                build_secrets=secrets,
                runtime=runtime,
                runtime_config=runtime_config,
                dry_run=dry_run,
            )
            payload = asdict(result)
            success = result.success
            if not dry_run and not wants_json(json_output):
                display_build_summary(
                    status_response=result.details,
                    registry_id=result.registry_id or "",
                    env_name=result.name,
                    console=HUDConsole(),
                )
        except Exception as exc:
            if not all_envs:
                raise
            error = map_exception(exc)
            payload = {"success": False, **error.to_payload()}
            success = False
            HUDConsole().error(f"{target.name}: {error.message}")
        (succeeded if success else failed).append(target.name)
        entries.append({"directory": target.name, **payload})
    if wants_json(json_output):
        emit_json(
            {"succeeded": succeeded, "failed": failed, "dry_run": dry_run, "environments": entries}
            if all_envs
            else entries[0]
        )
    elif dry_run:
        for entry in entries:
            HUDConsole().info(f"Would deploy {entry.get('name', entry['directory'])}")
            if entry.get("dotenv_pending"):
                HUDConsole().info("Uploading .env requires an explicit choice before deployment.")
    if failed:
        raise typer.Exit(1)
