"""Build HUD environments and generate lock files."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from hud.cli.utils.source_hash import compute_source_hash, list_source_files
from hud.clients import MCPClient
from hud.utils.hud_console import HUDConsole
from hud.version import __version__ as hud_version

from .utils.registry import save_to_registry


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
        with open(lock_path) as f:
            lock_data = yaml.safe_load(f)

        # Look for internal version in build metadata
        build_data = lock_data.get("build", {})
        return build_data.get("version", None)
    except Exception:
        return None


def get_docker_image_digest(image: str) -> str | None:
    """Get the digest of a Docker image."""
    try:
        result = subprocess.run(  # noqa: S603
            ["docker", "inspect", "--format", "{{.RepoDigests}}", image],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse the output - it's in format [repo@sha256:digest]
        digests = result.stdout.strip()
        if digests and digests != "[]":
            # Extract the first digest
            digest_list = eval(digests)  # noqa: S307 # Safe since it's from docker
            if digest_list:
                # Return full image reference with digest
                return digest_list[0]
    except Exception:  # noqa: S110
        # Don't print error here, let calling code handle it
        pass
    return None


def get_docker_image_id(image: str) -> str | None:
    """Get the ID of a Docker image."""
    try:
        result = subprocess.run(  # noqa: S603
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


def extract_env_vars_from_dockerfile(dockerfile_path: Path) -> tuple[list[str], list[str]]:
    """Extract required and optional environment variables from Dockerfile."""
    required = []
    optional = []

    if not dockerfile_path.exists():
        return required, optional

    # Parse both ENV and ARG directives
    content = dockerfile_path.read_text()
    arg_vars = set()  # Track ARG variables

    for line in content.splitlines():
        line = line.strip()

        # Look for ARG directives (build-time variables)
        if line.startswith("ARG "):
            parts = line[4:].strip().split("=", 1)
            var_name = parts[0].strip()
            if len(parts) == 1 or not parts[1].strip():
                # No default value = required
                arg_vars.add(var_name)
                if var_name not in required:
                    required.append(var_name)

        # Look for ENV directives (runtime variables)
        elif line.startswith("ENV "):
            parts = line[4:].strip().split("=", 1)
            var_name = parts[0].strip()

            # Check if it references an ARG variable (e.g., ENV MY_VAR=$MY_VAR)
            if len(parts) == 2 and parts[1].strip().startswith("$"):
                ref_var = parts[1].strip()[1:]
                if ref_var in arg_vars and var_name not in required:
                    required.append(var_name)
            elif len(parts) == 2 and not parts[1].strip():
                # No default value = required
                if var_name not in required:
                    required.append(var_name)
            elif len(parts) == 1:
                # No equals sign = required
                if var_name not in required:
                    required.append(var_name)

    return required, optional


async def analyze_mcp_environment(
    image: str, verbose: bool = False, env_vars: dict[str, str] | None = None
) -> dict[str, Any]:
    """Analyze an MCP environment to extract metadata."""
    hud_console = HUDConsole()
    env_vars = env_vars or {}

    # Build Docker command to run the image, injecting any provided env vars
    from hud.cli.utils.docker import build_env_flags

    docker_cmd = ["docker", "run", "--rm", "-i", *build_env_flags(env_vars), image]

    # Show full docker command being used for analysis
    hud_console.dim_info("Command:", " ".join(docker_cmd))

    # Create MCP config consistently with analyze helpers
    from hud.cli.analyze import parse_docker_command

    mcp_config = parse_docker_command(docker_cmd)

    # Initialize client and measure timing
    start_time = time.time()
    client = MCPClient(mcp_config=mcp_config, verbose=verbose, auto_trace=False)
    initialized = False

    try:
        if verbose:
            hud_console.info("Initializing MCP client...")

        # Add timeout to fail fast instead of hanging (60 seconds)
        await asyncio.wait_for(client.initialize(), timeout=60.0)
        initialized = True
        initialize_ms = int((time.time() - start_time) * 1000)

        # Delegate to standard analysis helper for consistency
        full_analysis = await client.analyze_environment()

        # Normalize to build's expected fields
        tools_list = full_analysis.get("tools", [])
        return {
            "initializeMs": initialize_ms,
            "toolCount": len(tools_list),
            "tools": tools_list,
            "success": True,
        }
    except TimeoutError:
        from hud.shared.exceptions import HudException

        hud_console.error("MCP server initialization timed out after 60 seconds")
        hud_console.info(
            "The server likely crashed during startup - check stderr logs with 'hud debug'"
        )
        raise HudException("MCP server initialization timeout") from None
    except Exception as e:
        from hud.shared.exceptions import HudException

        # Convert to HudException for better error messages and hints
        raise HudException from e
    finally:
        # Only shutdown if we successfully initialized
        if initialized:
            try:
                await client.shutdown()
            except Exception:
                # Ignore shutdown errors
                hud_console.warning("Failed to shutdown MCP client")


def build_docker_image(
    directory: Path,
    tag: str,
    no_cache: bool = False,
    verbose: bool = False,
    build_args: dict[str, str] | None = None,
    platform: str | None = None,
) -> bool:
    """Build a Docker image from a directory."""
    hud_console = HUDConsole()
    build_args = build_args or {}

    # Check if Dockerfile exists
    dockerfile = directory / "Dockerfile"
    if not dockerfile.exists():
        hud_console.error(f"No Dockerfile found in {directory}")
        return False

    # Default platform to match RL pipeline unless explicitly overridden
    effective_platform = platform if platform is not None else "linux/amd64"

    # Build command
    cmd = ["docker", "build"]
    if effective_platform:
        cmd.extend(["--platform", effective_platform])
    cmd.extend(["-t", tag])
    if no_cache:
        cmd.append("--no-cache")

    # Add build args
    for key, value in build_args.items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.append(str(directory))

    # Always show build output
    hud_console.info(f"Running: {' '.join(cmd)}")

    try:
        # Use Docker's native output formatting - no capture, let Docker handle display
        result = subprocess.run(cmd, check=False)  # noqa: S603
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
) -> None:
    """Build a HUD environment and generate lock file."""
    hud_console = HUDConsole()
    env_vars = env_vars or {}
    hud_console.header("HUD Environment Build")

    # Resolve directory
    env_dir = Path(directory).resolve()
    if not env_dir.exists():
        hud_console.error(f"Directory not found: {directory}")
        raise typer.Exit(1)

    from hud.cli.utils.docker import require_docker_running

    require_docker_running()

    # Step 1: Check for hud.lock.yaml (previous build)
    lock_path = env_dir / "hud.lock.yaml"
    base_name = None

    if lock_path.exists():
        try:
            with open(lock_path) as f:
                lock_data = yaml.safe_load(f)
            # Get base name from lock file (strip version/digest)
            lock_image = lock_data.get("images", {}).get("local") or lock_data.get("image", "")
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
        dockerfile_path = env_dir / "Dockerfile"
        if not dockerfile_path.exists():
            hud_console.error(f"Not a valid environment directory: {directory}")
            hud_console.info("Expected: Dockerfile or hud.lock.yaml")
            raise typer.Exit(1)

        # First build - use directory name
        base_name = env_dir.name
        hud_console.info(f"First build - using base name: {base_name}")

    # If user provides --tag, respect it; otherwise use base name only (version added later)
    if tag:
        # User explicitly provided a tag
        image_tag = tag
        base_name = image_tag.split(":")[0] if ":" in image_tag else image_tag
    else:
        # No tag provided - we'll add version later
        image_tag = None

    # Build temporary image first
    temp_tag = f"hud-build-temp:{int(time.time())}"

    hud_console.progress_message(f"Building Docker image: {temp_tag}")

    # Build the image (env vars are for runtime, not build time)
    if not build_docker_image(
        env_dir,
        temp_tag,
        no_cache,
        verbose,
        build_args=None,
        platform=platform,
    ):
        hud_console.error("Docker build failed")
        raise typer.Exit(1)

    hud_console.success(f"Built temporary image: {temp_tag}")

    # Analyze the environment (merge folder .env if present)
    hud_console.progress_message("Analyzing MCP environment...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Merge .env from env_dir for analysis only
        try:
            from hud.cli.utils.docker import load_env_vars_for_dir

            env_from_file = load_env_vars_for_dir(env_dir)
        except Exception:
            env_from_file = {}
        merged_env_for_analysis = {**env_from_file, **(env_vars or {})}

        analysis = loop.run_until_complete(
            analyze_mcp_environment(temp_tag, verbose, merged_env_for_analysis)
        )
    except Exception as e:
        hud_console.error(f"Failed to analyze MCP environment: {e}")
        hud_console.info("")
        hud_console.info("To debug this issue, run:")
        hud_console.command_example(f"hud debug {temp_tag}")
        hud_console.info("")
        raise typer.Exit(1) from e
    finally:
        loop.close()

    hud_console.success(f"Analyzed environment: {analysis['toolCount']} tools found")

    # Extract environment variables from Dockerfile
    dockerfile_path = env_dir / "Dockerfile"
    required_env, optional_env = extract_env_vars_from_dockerfile(dockerfile_path)

    # Merge user-provided env vars with detected ones
    provided_env_vars: dict[str, str] = {}
    missing_required = []
    if env_vars:
        # Use placeholders in lock file for any provided values to avoid storing secrets
        provided_env_vars = {k: f"${{{k}}}" for k in env_vars}
        # Track which required vars are still missing
        missing_required = [e for e in required_env if e not in env_vars]

        # Show what env vars were provided
        hud_console.success(f"Using provided environment variables: {', '.join(env_vars.keys())}")
    else:
        missing_required = required_env[:]

    # Warn about missing required variables
    if missing_required:
        hud_console.warning(
            f"Missing required environment variables: {', '.join(missing_required)}"
        )
        hud_console.info(
            "These can be added to the lock file after build or provided with -e flags"
        )

    # Check for existing version and increment
    lock_path = env_dir / "hud.lock.yaml"
    existing_version = get_existing_version(lock_path)

    if existing_version:
        # Increment existing version
        new_version = increment_version(existing_version)
        hud_console.info(f"Incrementing version: {existing_version} → {new_version}")
    else:
        # Start with 0.1.0 for new environments
        new_version = "0.1.0"
        hud_console.info(f"Setting initial version: {new_version}")

    # Determine base name for image references
    if image_tag:
        base_name = image_tag.split(":")[0] if ":" in image_tag else image_tag

    # Create lock file content with images subsection at top
    lock_content = {
        "version": "1.1",  # Lock file format version
        "images": {
            "local": f"{base_name}:{new_version}",  # Local tag with version
            "full": None,  # Will be set with digest after build
            "pushed": None,  # Will be set by hud push
        },
        "build": {
            "generatedAt": datetime.now(UTC).isoformat() + "Z",
            "hudVersion": hud_version,
            "directory": str(env_dir.name),
            "version": new_version,
            # Fast source fingerprint for change detection
            "sourceHash": compute_source_hash(env_dir),
        },
        "environment": {
            "initializeMs": analysis["initializeMs"],
            "toolCount": analysis["toolCount"],
        },
    }

    # Add environment variables section if any exist
    if missing_required or optional_env or provided_env_vars:
        lock_content["environment"]["variables"] = {}

        # Add note about editing environment variables
        lock_content["environment"]["variables"]["_note"] = (
            "You can edit this section to add or modify environment variables. "
            "Provided variables will be used when running the environment."
        )

        if provided_env_vars:
            lock_content["environment"]["variables"]["provided"] = provided_env_vars
        if missing_required:
            lock_content["environment"]["variables"]["required"] = missing_required
        if optional_env:
            lock_content["environment"]["variables"]["optional"] = optional_env

    # Add tools with full schemas for RL config generation
    if analysis["tools"]:
        lock_content["tools"] = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            for tool in analysis["tools"]
        ]

    # Write lock file
    lock_path = env_dir / "hud.lock.yaml"
    with open(lock_path, "w") as f:
        yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)

    # Also write the file list we hashed for transparency (non-essential)
    with contextlib.suppress(Exception):
        files = [
            str(p.resolve().relative_to(env_dir)).replace("\\", "/")
            for p in list_source_files(env_dir)
        ]
        lock_content["build"]["sourceFiles"] = files
        with open(lock_path, "w") as f:
            yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)

    hud_console.success("Created lock file: hud.lock.yaml")

    # Calculate lock file hash
    lock_content_str = yaml.dump(lock_content, default_flow_style=False, sort_keys=True)
    lock_hash = hashlib.sha256(lock_content_str.encode()).hexdigest()
    lock_size = len(lock_content_str)

    # Rebuild with label containing lock file hash
    hud_console.progress_message("Rebuilding with lock file metadata...")

    # Build final image with label (uses cache from first build)
    # Create tags: versioned and latest (and custom tag if provided)
    version_tag = f"{base_name}:{new_version}"
    latest_tag = f"{base_name}:latest"

    label_cmd = ["docker", "build"]
    # Use same defaulting for the second build step
    label_platform = platform if platform is not None else "linux/amd64"
    if label_platform:
        label_cmd.extend(["--platform", label_platform])
    label_cmd.extend(
        [
            "--label",
            f"org.hud.manifest.head={lock_hash}:{lock_size}",
            "--label",
            f"org.hud.version={new_version}",
            "-t",
            version_tag,  # Always tag with new version
            "-t",
            latest_tag,  # Always tag with latest
        ]
    )

    # Add custom tag if user provided one
    if image_tag and image_tag not in [version_tag, latest_tag]:
        label_cmd.extend(["-t", image_tag])

    label_cmd.append(str(env_dir))

    # Run rebuild using Docker's native output formatting
    if verbose:
        # Show Docker's native output when verbose
        result = subprocess.run(label_cmd, check=False)  # noqa: S603
    else:
        # Capture output for error reporting, but don't show unless it fails
        result = subprocess.run(  # noqa: S603
            label_cmd, capture_output=True, text=True, check=False
        )

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

    # NOW get the image ID after the final build
    image_id = get_docker_image_id(version_tag)
    if image_id:
        # Store full reference with digest
        if image_id.startswith("sha256:"):
            lock_content["images"]["full"] = f"{version_tag}@{image_id}"
        else:
            lock_content["images"]["full"] = f"{version_tag}@sha256:{image_id}"

        # Update the lock file with the full image reference
        with open(lock_path, "w") as f:
            yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)

        hud_console.success("Updated lock file with image digest")
    else:
        hud_console.warning("Could not retrieve image digest")

    # Remove temp image after we're done
    subprocess.run(["docker", "rmi", "-f", temp_tag], capture_output=True)  # noqa: S603, S607

    # Add to local registry
    if image_id:
        # Save to local registry using the helper
        local_ref = lock_content.get("images", {}).get("local", version_tag)
        save_to_registry(lock_content, local_ref, verbose)

    # Print summary
    hud_console.section_title("Build Complete")

    # Show the version tag as primary since that's what will be pushed
    hud_console.status_item("Built image", version_tag, primary=True)

    # Show additional tags
    additional_tags = [latest_tag]
    if image_tag and image_tag not in [version_tag, latest_tag]:
        additional_tags.append(image_tag)
    hud_console.status_item("Also tagged", ", ".join(additional_tags))

    hud_console.status_item("Version", new_version)
    hud_console.status_item("Lock file", "hud.lock.yaml")
    hud_console.status_item("Tools found", str(analysis["toolCount"]))

    # Show the digest info separately if we have it
    if image_id:
        hud_console.dim_info("\nImage digest", image_id)

    hud_console.section_title("Next Steps")
    hud_console.info("Test locally:")
    hud_console.command_example("hud dev", "Hot-reload development")
    hud_console.command_example(f"hud run {latest_tag}", "Run the built image")
    hud_console.info("")
    hud_console.info("Publish to registry:")
    hud_console.command_example("hud push", f"Push as {version_tag}")
    hud_console.command_example("hud push --tag latest", "Push with custom tag")
    hud_console.info("")
    hud_console.info("The lock file can be used to reproduce this exact environment.")


def build_command(
    directory: str = typer.Argument(".", help="Environment directory to build"),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Docker image tag (default: from pyproject.toml)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without Docker cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    env_vars: dict[str, str] | None = None,
    platform: str | None = None,
) -> None:
    """Build a HUD environment and generate lock file."""
    build_environment(directory, tag, no_cache, verbose, env_vars, platform)
