"""Build HUD environments and generate lock files."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

from hud.cli.utils.environment import extract_env_vars_from_dockerfile, find_dockerfile
from hud.cli.utils.lockfile import (
    build_lock_data,
    dump_lock_data,
)
from hud.shared.hints import render_hints, secrets_in_build_args
from hud.utils.hud_console import HUDConsole


def _read_env_manifest(env_dir: Path) -> dict[str, Any]:
    """Read a v6 environment's manifest (capabilities + tasks) from its source.

    Imports ``env.py`` from *env_dir* and returns ``Environment.to_dict()`` — the
    declarative manifest (name, version, capabilities, tasks) baked into the lock.
    No container run is needed: the manifest is declared, not introspected.
    """
    from hud.environment import Environment
    from hud.eval import load_module

    env_file = env_dir / "env.py"
    if not env_file.exists():
        raise FileNotFoundError(f"no env.py found in {env_dir}")
    module = load_module(env_file)
    envs = [v for v in vars(module).values() if isinstance(v, Environment)]
    if not envs:
        raise ValueError(f"no Environment instance defined in {env_file}")
    if len(envs) > 1:
        raise ValueError(f"multiple Environments in {env_file}; expected exactly one")
    return envs[0].to_dict()


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
        from hud.cli.utils.lockfile import load_lock

        lock_data = load_lock(lock_path)
        return lock_data.get("build", {}).get("version", None)
    except Exception:
        return None


def _redacted_cmd(cmd: list[str]) -> str:
    """Render a docker command for logging with --build-arg/--secret values masked.

    Build args and secrets can carry credentials; keep the arg key for readability
    but never echo its value (or a secret spec) into logs.
    """
    parts: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            key = token.split("=", 1)[0]
            parts.append(f"{key}=***" if "=" in token else "***")
            redact_next = False
        elif token in ("--build-arg", "--secret"):
            parts.append(token)
            redact_next = True
        else:
            parts.append(token)
    return " ".join(parts)


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


def build_docker_image(
    directory: Path,
    tag: str,
    no_cache: bool = False,
    verbose: bool = False,
    build_args: dict[str, str] | None = None,
    platform: str | None = None,
    secrets: list[str] | None = None,
    docker_args: list[str] | None = None,
) -> bool:
    """Build a Docker image from a directory.

    Wraps ``docker buildx build``. Any flags that Docker understands
    (``--cache-from``, ``--push``, ``--load``, etc.) belong in *docker_args*
    and are appended to the command as-is. Unless the caller explicitly picks
    an output mode, the result is loaded into the host daemon for local
    analysis/debugging.
    """
    hud_console = HUDConsole()
    build_args = build_args or {}
    secrets = secrets or []
    docker_args = docker_args or []

    dockerfile = find_dockerfile(directory)
    if dockerfile is None:
        hud_console.error(f"No Dockerfile found in {directory}")
        hud_console.info("Expected: Dockerfile.hud or Dockerfile")
        return False

    effective_platform = platform if platform is not None else "linux/amd64"
    cmd = ["docker", "buildx", "build"]

    if dockerfile.name != "Dockerfile":
        cmd.extend(["-f", str(dockerfile)])

    if effective_platform:
        cmd.extend(["--platform", effective_platform])
    cmd.extend(["-t", tag])
    if no_cache:
        cmd.append("--no-cache")

    # Passthrough: cache, push, and any other Docker-native flags
    cmd.extend(docker_args)

    # Local hud build expects a daemon-loaded image unless the caller explicitly
    # selects another buildx output mode such as --push/--output.
    if not _has_build_output_arg(docker_args):
        cmd.append("--load")

    for key, value in build_args.items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    for secret in secrets:
        cmd.extend(["--secret", secret])

    cmd.append(str(directory))

    hud_console.info(f"Running: {_redacted_cmd(cmd)}")

    try:
        env = os.environ.copy()
        if secrets:
            env["DOCKER_BUILDKIT"] = "1"
        result = subprocess.run(cmd, check=False, env=env)
        return result.returncode == 0
    except Exception as e:
        hud_console.error(f"Build error: {e}")
        return False


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
    if not env_dir.exists():
        hud_console.error(f"Directory not found: {directory}")
        raise typer.Exit(1)

    from hud.cli.utils.docker import require_docker_running

    require_docker_running()

    # Step 1: Check for hud.lock.yaml (previous build)
    from hud.cli.utils.lockfile import LOCK_FILENAME, get_local_image, load_lock

    lock_path = env_dir / LOCK_FILENAME
    base_name = None

    if lock_path.exists():
        try:
            lock_data = load_lock(lock_path)
            lock_image = get_local_image(lock_data)
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
        dockerfile_path = find_dockerfile(env_dir)
        if dockerfile_path is None:
            hud_console.error(f"Not a valid environment directory: {directory}")
            hud_console.info("Expected: Dockerfile.hud, Dockerfile, or hud.lock.yaml")
            raise typer.Exit(1)

        # First build - use directory name
        base_name = env_dir.name
        hud_console.info(f"First build - using base name: {base_name}")
        if dockerfile_path.name == "Dockerfile.hud":
            hud_console.info("Using Dockerfile.hud")

    # If user provides --tag, respect it; otherwise use base name only (version added later)
    if tag:
        # User explicitly provided a tag
        image_tag = tag
        base_name = image_tag.split(":")[0] if ":" in image_tag else image_tag
    else:
        # No tag provided - we'll add version later
        image_tag = None

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

    # Set up build tags
    if pushing:
        if not tag:
            hud_console.error("--push requires --tag with a registry-qualified image name")
            raise typer.Exit(1)
        build_tag = tag
        hud_console.progress_message("Building and pushing Docker image...")
    else:
        build_tag = f"hud-build-temp:{int(time.time())}"
        hud_console.progress_message(f"Building Docker image: {build_tag}")

    # Build the image (env vars are for runtime, not build time)
    if not build_docker_image(
        env_dir,
        build_tag,
        no_cache,
        verbose,
        build_args=build_args or None,
        platform=platform,
        secrets=secrets,
        docker_args=docker_args,
    ):
        hud_console.error("Docker build failed")
        raise typer.Exit(1)

    # Get image locally for analysis
    if pushing:
        hud_console.success(f"Pushed image: {build_tag}")
        hud_console.progress_message("Pulling image for analysis...")
        pull_result = subprocess.run(
            ["docker", "pull", build_tag],  # noqa: S607
            check=False,
        )
        if pull_result.returncode != 0:
            hud_console.error(f"Failed to pull image: {build_tag}")
            raise typer.Exit(1)
        analysis_image = build_tag
    else:
        analysis_image = build_tag
        hud_console.success(f"Built temporary image: {build_tag}")

    # Load .env from env_dir (used for env-var requirements in the lock).
    try:
        from hud.cli.utils.docker import load_env_vars_for_dir

        env_from_file = load_env_vars_for_dir(env_dir)
    except Exception:
        env_from_file = {}

    # Read the v6 environment manifest (capabilities + tasks) from the env source.
    hud_console.progress_message("Reading environment manifest...")
    try:
        analysis = _read_env_manifest(env_dir)
    except Exception as e:
        hud_console.error(f"Failed to read environment manifest: {e}")
        raise typer.Exit(1) from e

    cap_count = len(analysis.get("capabilities") or [])
    task_count = len(analysis.get("tasks") or [])
    hud_console.success(f"Environment manifest: {cap_count} capability(ies), {task_count} task(s)")

    # Extract environment variables from Dockerfile
    dockerfile_path = find_dockerfile(env_dir) or env_dir / "Dockerfile"
    required_env, _optional_env = extract_env_vars_from_dockerfile(dockerfile_path)

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

    # Determine base name for image references
    if image_tag:
        base_name = image_tag.split(":")[0] if ":" in image_tag else image_tag

    effective_platform = platform if platform is not None else "linux/amd64"

    env_vars_from_file = set(env_from_file.keys()) if env_from_file else set()
    lock_content = build_lock_data(
        source_dir=env_dir,
        analysis=analysis,
        version=new_version,
        image_name=base_name,
        full_image_ref=None,
        pushed_image_ref=build_tag if pushing else None,
        env_vars=env_vars or None,
        additional_required_env_vars=env_vars_from_file,
        platform=effective_platform,
        local_image_ref=build_tag if pushing else None,
    )

    # Write lock file
    lock_path = env_dir / "hud.lock.yaml"
    with open(lock_path, "w") as f:
        f.write(dump_lock_data(lock_content))

    hud_console.success("Created lock file: hud.lock.yaml")

    # Calculate lock file hash
    lock_content_str = dump_lock_data(lock_content, sort_keys=True)
    lock_hash = hashlib.sha256(lock_content_str.encode()).hexdigest()
    lock_size = len(lock_content_str)

    version_tag = f"{base_name}:{new_version}"
    latest_tag = f"{base_name}:latest"

    if pushing:
        # Image already pushed — get digest from pulled image
        image_id = get_docker_image_id(analysis_image)
        if image_id:
            if image_id.startswith("sha256:"):
                lock_content["images"]["full"] = f"{analysis_image}@{image_id}"
            else:
                lock_content["images"]["full"] = f"{analysis_image}@sha256:{image_id}"
            with open(lock_path, "w") as f:
                f.write(dump_lock_data(lock_content))
            hud_console.success("Updated lock file with image digest")
        else:
            hud_console.warning("Could not retrieve image digest")
        subprocess.run(["docker", "rmi", "-f", analysis_image], capture_output=True)  # noqa: S607
    else:
        # Rebuild with label containing lock file hash
        hud_console.progress_message("Rebuilding with lock file metadata...")

        # Reuse Docker flags for the label rebuild, but never --push.
        label_docker_args = [a for a in (docker_args or []) if a != "--push"]
        label_cmd = ["docker", "buildx", "build"]

        if dockerfile_path and dockerfile_path.name != "Dockerfile":
            label_cmd.extend(["-f", str(dockerfile_path)])

        label_platform = platform if platform is not None else "linux/amd64"
        if label_platform:
            label_cmd.extend(["--platform", label_platform])

        label_cmd.extend(label_docker_args)
        if not _has_build_output_arg(label_docker_args):
            label_cmd.append("--load")

        label_cmd.extend(
            [
                "--label",
                f"org.hud.manifest.head={lock_hash}:{lock_size}",
                "--label",
                f"org.hud.version={new_version}",
                "-t",
                version_tag,
                "-t",
                latest_tag,
            ]
        )

        if image_tag and image_tag not in [version_tag, latest_tag]:
            label_cmd.extend(["-t", image_tag])

        for key, value in build_args.items():
            label_cmd.extend(["--build-arg", f"{key}={value}"])

        for secret in secrets or []:
            label_cmd.extend(["--secret", secret])

        label_cmd.append(str(env_dir))

        env = os.environ.copy()
        if secrets:
            env["DOCKER_BUILDKIT"] = "1"
        if verbose:
            result = subprocess.run(label_cmd, check=False, env=env)
        else:
            result = subprocess.run(label_cmd, capture_output=True, text=True, check=False, env=env)

        if result.returncode != 0:
            hud_console.error("Failed to rebuild with label")
            if not verbose and result.stderr:
                hud_console.info("Error output:")
                hud_console.info(str(result.stderr))
            if not verbose:
                hud_console.info("")
                hud_console.info("Run with --verbose to see full build output:")
                hud_console.command_example("hud build --verbose")
            raise typer.Exit(1)

        hud_console.success("Built final image with lock file metadata")

        image_id = get_docker_image_id(version_tag)
        if image_id:
            if image_id.startswith("sha256:"):
                lock_content["images"]["full"] = f"{version_tag}@{image_id}"
            else:
                lock_content["images"]["full"] = f"{version_tag}@sha256:{image_id}"
            with open(lock_path, "w") as f:
                f.write(dump_lock_data(lock_content))
            hud_console.success("Updated lock file with image digest")
        else:
            hud_console.warning("Could not retrieve image digest")

        subprocess.run(["docker", "rmi", "-f", build_tag], capture_output=True)  # noqa: S607

    # Print summary
    hud_console.section_title("Build Complete")

    if pushing:
        hud_console.status_item("Pushed image", build_tag, primary=True)
    else:
        hud_console.status_item("Built image", version_tag, primary=True)
        additional_tags = [latest_tag]
        if image_tag and image_tag not in [version_tag, latest_tag]:
            additional_tags.append(image_tag)
        hud_console.status_item("Also tagged", ", ".join(additional_tags))

    hud_console.status_item("Version", new_version)
    hud_console.status_item("Lock file", "hud.lock.yaml")
    hud_console.status_item("Tasks", str(task_count))
    hud_console.status_item("Capabilities", str(cap_count))

    if image_id:
        hud_console.dim_info("\nImage digest", image_id)

    hud_console.section_title("Next Steps")
    if pushing:
        hud_console.info("Test the pushed image:")
        hud_console.command_example(f"hud debug {build_tag}", "Test MCP compliance")
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
