"""Build HUD environments and generate lock files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import typer

from hud.environment import lock
from hud.environment.source import EnvironmentSource
from hud.shared.hints import render_hints, secrets_in_build_args
from hud.utils.hud_console import HUDConsole


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse version string like '1.0.0' or '1.0' into tuple of integers."""
    # Remove 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Split by dots and pad with zeros if needed
    parts = version_str.split(".")
    parts.extend(["0"] * (3 - len(parts)))  # Ensure we have at least 3 parts

    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        # Default to 0.0.0 if parsing fails
        return (0, 0, 0)


def increment_version(version_str: str, increment_type: str = "patch") -> str:
    """Increment version string. increment_type can be 'major', 'minor', or 'patch'."""
    major, minor, patch = parse_version(version_str)

    if increment_type == "major":
        return f"{major + 1}.0.0"
    elif increment_type == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


def get_existing_version(lock_path: Path) -> str | None:
    """Get the internal version from existing lock file if it exists."""
    if not lock_path.exists():
        return None

    try:
        lock_data = lock.read_lock(lock_path)
        return lock_data.get("build", {}).get("version", None)
    except Exception:
        return None


def get_docker_image_id(image: str) -> str | None:
    """Get the ID of a Docker image."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", image],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        image_id = result.stdout.strip()
        if image_id:
            return image_id
        return None
    except Exception:
        # Don't log here to avoid import issues
        return None


def _image_ref_with_digest(image_ref: str) -> tuple[str | None, str | None]:
    image_id = get_docker_image_id(image_ref)
    if not image_id:
        return None, None
    digest = image_id if image_id.startswith("sha256:") else f"sha256:{image_id}"
    return f"{image_ref}@{digest}", image_id


def check_dockerfile_for_secrets(directory: Path, dockerfile: Path) -> list[str]:
    """Run docker buildx build --check to detect secrets in ARG/ENV.

    Returns a list of variable names that were flagged as potential secrets.
    This is a fast, non-building lint check.
    """
    hud_console = HUDConsole()

    cmd = ["docker", "buildx", "build", "--check"]
    if dockerfile.name != "Dockerfile":
        cmd.extend(["-f", str(dockerfile)])
    cmd.append(str(directory))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr

        pattern = r'Do not use ARG or ENV instructions for sensitive data \((ARG|ENV) "([^"]+)"\)'
        matches = re.findall(pattern, output)

        if matches:
            secret_vars = [f"{var_type} {var_name}" for var_type, var_name in matches]
            return secret_vars

    except subprocess.TimeoutExpired:
        hud_console.warning("Dockerfile check timed out")
    except Exception as e:
        hud_console.debug(f"Dockerfile secrets check failed: {e}")

    return []


def display_secrets_warning(secret_vars: list[str]) -> None:
    """Display a warning about secrets found in Dockerfile ARG/ENV."""

    hud_console = HUDConsole()
    hud_console.print("")
    render_hints([secrets_in_build_args(secret_vars)])
    hud_console.print("")


def _has_build_output_arg(docker_args: list[str]) -> bool:
    """Return True when *docker_args* already choose a build output mode."""
    return any(
        arg in ("--push", "--load", "--output", "-o") or arg.startswith(("--output=", "-o="))
        for arg in docker_args
    )


def _has_non_daemon_output(docker_args: list[str]) -> bool:
    """Return True when *docker_args* route build output away from the local daemon.

    Detects ``--output``/``-o`` without an accompanying ``--load``, meaning
    the built image won't be available for local analysis.
    """
    has_custom = any(
        arg in ("--output", "-o") or arg.startswith(("--output=", "-o=")) for arg in docker_args
    )
    return has_custom and "--load" not in docker_args


def _docker_buildx_cmd(
    directory: Path,
    dockerfile: Path,
    *,
    tags: list[str],
    labels: dict[str, str] | None = None,
    no_cache: bool = False,
    platform: str | None = None,
    build_args: dict[str, str] | None = None,
    secrets: list[str] | None = None,
    docker_args: list[str] | None = None,
) -> list[str]:
    cmd = ["docker", "buildx", "build"]
    if dockerfile.name != "Dockerfile":
        cmd.extend(["-f", str(dockerfile)])
    if platform:
        cmd.extend(["--platform", platform])
    for tag in tags:
        cmd.extend(["-t", tag])
    if no_cache:
        cmd.append("--no-cache")

    passthrough = docker_args or []
    cmd.extend(passthrough)
    if not _has_build_output_arg(passthrough):
        cmd.append("--load")

    for key, value in (labels or {}).items():
        cmd.extend(["--label", f"{key}={value}"])
    for key, value in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{key}={value}"])
    for secret in secrets or []:
        cmd.extend(["--secret", secret])

    cmd.append(str(directory))
    return cmd


def _docker_env(secrets: list[str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if secrets:
        env["DOCKER_BUILDKIT"] = "1"
    return env


def _restore_lock(lock_path: Path, previous: str | None) -> None:
    if previous is None:
        lock_path.unlink(missing_ok=True)
    else:
        lock_path.write_text(previous, encoding="utf-8")


def build_environment(
    directory: str = ".",
    tag: str | None = None,
    no_cache: bool = False,
    verbose: bool = False,
    env_vars: dict[str, str] | None = None,
    platform: str | None = None,
    secrets: list[str] | None = None,
    build_args: dict[str, str] | None = None,
    docker_args: list[str] | None = None,
) -> None:
    """Build a HUD environment and generate lock file."""
    hud_console = HUDConsole()
    env_vars = env_vars or {}
    build_args = build_args or {}
    hud_console.header("HUD Environment Build")

    # Resolve directory
    env_dir = Path(directory).resolve()
    env_source = EnvironmentSource.open(env_dir)
    if not env_dir.exists():
        hud_console.error(f"Directory not found: {directory}")
        raise typer.Exit(1)

    from hud.cli.utils.docker import require_docker_running

    require_docker_running()

    # Step 1: Check for hud.lock.yaml (previous build)
    lock_path = env_source.lock_path
    base_name = None

    if lock_path.exists():
        try:
            lock_data = lock.read_lock(lock_path)
            lock_image = lock.local_image(lock_data)
            if lock_image:
                # Remove @sha256:... digest if present
                if "@" in lock_image:
                    lock_image = lock_image.split("@")[0]
                # Extract base name (remove :version tag)
                base_name = lock_image.split(":")[0] if ":" in lock_image else lock_image
                hud_console.info(f"Using base name from lock file: {base_name}")
        except Exception as e:
            hud_console.warning(f"Could not read lock file: {e}")

    # Step 2: If no lock, check for Dockerfile
    if not base_name:
        dockerfile_path = env_source.dockerfile
        if dockerfile_path is None:
            hud_console.error(f"Not a valid environment directory: {directory}")
            hud_console.info("Expected: Dockerfile.hud, Dockerfile, or hud.lock.yaml")
            raise typer.Exit(1)

        # First build - use directory name
        base_name = env_dir.name
        hud_console.info(f"First build - using base name: {base_name}")
        if dockerfile_path.name == "Dockerfile.hud":
            hud_console.info("Using Dockerfile.hud")

    if tag:
        base_name = tag.split(":")[0] if ":" in tag else tag

    # Compute version before building (needed for image tags when --push is used)
    existing_version = get_existing_version(lock_path)
    if existing_version:
        new_version = increment_version(existing_version)
        hud_console.info(f"Incrementing version: {existing_version} → {new_version}")
    else:
        new_version = "0.1.0"
        hud_console.info(f"Setting initial version: {new_version}")

    # Detect --push in docker passthrough args
    pushing = "--push" in (docker_args or [])

    if not pushing and _has_non_daemon_output(docker_args or []):
        hud_console.error(
            "A custom --output was specified without --load; "
            "the image would not be available in the local Docker daemon for analysis."
        )
        hud_console.info("Add --load alongside your --output flag, or use --push instead.")
        raise typer.Exit(1)

    if pushing and not tag:
        hud_console.error("--push requires --tag with a registry-qualified image name")
        raise typer.Exit(1)

    try:
        from hud.cli.utils.docker import load_env_vars_for_dir

        env_from_file = load_env_vars_for_dir(env_dir)
    except Exception:
        env_from_file = {}

    # Read the v6 environment manifest (capabilities + tasks) from the env source.
    hud_console.progress_message("Reading environment manifest...")
    try:
        analysis = env_source.manifest()
    except Exception as e:
        hud_console.error(f"Failed to read environment manifest: {e}")
        raise typer.Exit(1) from e

    cap_count = len(analysis.get("capabilities") or [])
    task_count = len(analysis.get("tasks") or [])
    hud_console.success(f"Environment manifest: {cap_count} capability(ies), {task_count} task(s)")

    dockerfile_path = env_source.dockerfile
    if dockerfile_path is None:
        hud_console.error(f"Not a valid environment directory: {directory}")
        hud_console.info("Expected: Dockerfile.hud, Dockerfile, or hud.lock.yaml")
        raise typer.Exit(1)
    required_env = env_source.dockerfile_env_vars()

    # Show env vars detected from .env file
    if env_from_file:
        hud_console.info(
            f"Detected environment variables from .env file: {', '.join(sorted(env_from_file.keys()))}"  # noqa: E501
        )

    # Create a complete set of all required variables for warning
    all_required_for_warning = set(required_env)
    all_required_for_warning.update(env_from_file.keys())

    # Find which ones are missing (not provided via -e flags)
    all_missing = all_required_for_warning - set(env_vars.keys() if env_vars else [])

    if all_missing:
        hud_console.warning(
            f"Environment variables not provided via -e flags: {', '.join(sorted(all_missing))}"
        )
        hud_console.info("These will be added to the required list in the lock file")

    # Check for secrets in ARG/ENV instructions
    secret_vars = check_dockerfile_for_secrets(env_dir, dockerfile_path)
    if secret_vars:
        display_secrets_warning(secret_vars)

    effective_platform = platform if platform is not None else "linux/amd64"
    version_tag = f"{base_name}:{new_version}"
    latest_tag = f"{base_name}:latest"
    if pushing:
        assert tag is not None
        primary_tag = tag
    else:
        primary_tag = version_tag

    lock_content = lock.build_lock_data(
        env_source,
        analysis=analysis,
        version=new_version,
        local_image_ref=primary_tag if pushing else version_tag,
        pushed_image_ref=primary_tag if pushing else None,
        env_vars=env_vars or None,
        extra_required_env=env_from_file.keys(),
        platform=effective_platform,
    )

    previous_lock = lock_path.read_text(encoding="utf-8") if lock_path.exists() else None
    lock.write_lock(lock_path, lock_content)
    hud_console.success("Created lock file: hud.lock.yaml")

    lock_hash, lock_size = lock.lock_fingerprint(lock_content)
    tags = [primary_tag] if pushing else [version_tag, latest_tag]
    if tag and tag not in tags:
        tags.append(tag)
    labels = (
        {}
        if pushing
        else {
            "org.hud.manifest.head": f"{lock_hash}:{lock_size}",
            "org.hud.version": new_version,
        }
    )

    build_cmd = _docker_buildx_cmd(
        env_dir,
        dockerfile_path,
        tags=tags,
        labels=labels,
        no_cache=no_cache,
        platform=effective_platform,
        build_args=build_args,
        secrets=secrets,
        docker_args=docker_args,
    )
    hud_console.progress_message(
        f"{'Building and pushing' if pushing else 'Building'} Docker image: {primary_tag}"
    )
    hud_console.info(f"Running: {' '.join(build_cmd)}")

    if verbose:
        result = subprocess.run(build_cmd, check=False, env=_docker_env(secrets))
    else:
        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            check=False,
            env=_docker_env(secrets),
        )

    if result.returncode != 0:
        _restore_lock(lock_path, previous_lock)
        hud_console.error("Docker build failed")
        if not verbose and result.stderr:
            hud_console.info("Error output:")
            hud_console.info(str(result.stderr))
        if not verbose:
            hud_console.info("")
            hud_console.info("Run with --verbose to see full build output:")
            hud_console.command_example("hud build --verbose")
        raise typer.Exit(1)

    if pushing:
        hud_console.success(f"Pushed image: {primary_tag}")
        hud_console.progress_message("Pulling image for digest...")
        pull_result = subprocess.run(["docker", "pull", primary_tag], check=False)  # noqa: S607
        if pull_result.returncode != 0:
            _restore_lock(lock_path, previous_lock)
            hud_console.error(f"Failed to pull image: {primary_tag}")
            raise typer.Exit(1)
        full_ref, image_id = _image_ref_with_digest(primary_tag)
        subprocess.run(["docker", "rmi", "-f", primary_tag], capture_output=True)  # noqa: S607
    else:
        hud_console.success("Built image with lock file metadata")
        full_ref, image_id = _image_ref_with_digest(version_tag)

    if full_ref:
        lock_content["images"]["full"] = full_ref
        lock.write_lock(lock_path, lock_content)
        hud_console.success("Updated lock file with image digest")
    else:
        hud_console.warning("Could not retrieve image digest")

    # Print summary
    hud_console.section_title("Build Complete")

    if pushing:
        hud_console.status_item("Pushed image", primary_tag, primary=True)
    else:
        hud_console.status_item("Built image", version_tag, primary=True)
        additional_tags = [latest_tag]
        if tag and tag not in [version_tag, latest_tag]:
            additional_tags.append(tag)
        hud_console.status_item("Also tagged", ", ".join(additional_tags))

    hud_console.status_item("Version", new_version)
    hud_console.status_item("Lock file", "hud.lock.yaml")
    hud_console.status_item("Tasks found", str(len(analysis.get("tasks") or [])))
    hud_console.status_item("Capabilities found", str(len(analysis.get("capabilities") or [])))

    if image_id:
        hud_console.dim_info("\nImage digest", image_id)

    hud_console.section_title("Next Steps")
    if pushing:
        hud_console.info("Test the pushed image:")
        hud_console.command_example(f"hud debug {primary_tag}", "Test MCP compliance")
    else:
        hud_console.info("Test locally:")
        hud_console.command_example("hud dev", "Hot-reload development")
        hud_console.command_example(f"hud debug {version_tag}", "Test MCP compliance")
    hud_console.info("")
    hud_console.info("Deploy to platform:")
    hud_console.command_example("hud deploy", "Build remotely and deploy")
    hud_console.info("")
    hud_console.info("The lock file can be used to reproduce this exact environment.")


def build_command(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Environment directory followed by optional arguments (e.g., '. -e API_KEY=secret')",
    ),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Docker image tag (default: from pyproject.toml)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without Docker cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    platform: str | None = typer.Option(
        None, "--platform", help="Set Docker target platform (e.g., linux/amd64)"
    ),
    secrets: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--secret",
        help=("Docker build secret (repeatable), e.g. --secret id=GITHUB_TOKEN,env=GITHUB_TOKEN"),
    ),
) -> None:
    """🏗️ Build a HUD environment and generate lock file.

    [not dim]This command:
    - Builds a Docker image from your environment
    - Analyzes the MCP server to extract metadata
    - Generates a hud.lock.yaml file for reproducibility

    Docker flags (--cache-from, --push, etc.) can be passed after --.

    Examples:
        hud build                    # Build current directory
        hud build environments/text_2048 -e API_KEY=secret
        hud build . --tag my-env:v1.0 -e VAR1=value1 -e VAR2=value2
        hud build . --no-cache       # Force rebuild
        hud build . --build-arg NODE_ENV=production  # Pass Docker build args
        hud build . --secret id=MY_KEY,env=MY_KEY  # Pass build secrets
        hud build . --push                         # Push to registry after build[/not dim]
    """
    if params:
        directory = params[0]
        extra_args = params[1:] if len(params) > 1 else []
    else:
        directory = "."
        extra_args = []

    from hud.cli.utils.args import split_docker_passthrough

    env_vars, build_args, docker_args = split_docker_passthrough(extra_args)

    build_environment(
        directory,
        tag,
        no_cache,
        verbose,
        env_vars or None,
        platform,
        secrets,
        build_args=build_args or None,
        docker_args=docker_args or None,
    )
