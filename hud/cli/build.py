"""Build HUD environments and generate lock files."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

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

    # Build Docker command to run the image
    docker_cmd = ["docker", "run", "--rm", "-i"]

    # Add environment variables
    for key, value in env_vars.items():
        docker_cmd.extend(["-e", f"{key}={value}"])

    docker_cmd.append(image)

    # Create MCP config
    config = {
        "server": {"command": docker_cmd[0], "args": docker_cmd[1:] if len(docker_cmd) > 1 else []}
    }

    # Initialize client and measure timing
    start_time = time.time()
    client = MCPClient(mcp_config=config, verbose=verbose, auto_trace=False)
    initialized = False

    try:
        if verbose:
            hud_console.info(f"Initializing MCP client with command: {' '.join(docker_cmd)}")

        await client.initialize()
        initialized = True
        initialize_ms = int((time.time() - start_time) * 1000)

        # Get tools
        tools = await client.list_tools()

        # Extract tool information
        tool_info = []
        for tool in tools:
            tool_dict = {"name": tool.name, "description": tool.description}
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                tool_dict["inputSchema"] = tool.inputSchema
            tool_info.append(tool_dict)

        return {
            "initializeMs": initialize_ms,
            "toolCount": len(tools),
            "tools": tool_info,
            "success": True,
        }
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
) -> bool:
    """Build a Docker image from a directory."""
    hud_console = HUDConsole()
    build_args = build_args or {}

    # Check if Dockerfile exists
    dockerfile = directory / "Dockerfile"
    if not dockerfile.exists():
        hud_console.error(f"No Dockerfile found in {directory}")
        return False

    # Build command
    cmd = ["docker", "build", "-t", tag]
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

    # Check for pyproject.toml
    pyproject_path = env_dir / "pyproject.toml"
    if not pyproject_path.exists():
        hud_console.error(f"No pyproject.toml found in {directory}")
        raise typer.Exit(1)

    # Read pyproject.toml to get image name
    try:
        import toml

        pyproject = toml.load(pyproject_path)
        default_image = pyproject.get("tool", {}).get("hud", {}).get("image", None)
        if not default_image:
            # Generate default from directory name
            default_image = f"{env_dir.name}:dev"
    except Exception:
        default_image = f"{env_dir.name}:dev"

    # Use provided tag or default
    if not tag:
        tag = default_image

    # Build temporary image first
    temp_tag = f"hud-build-temp:{int(time.time())}"

    hud_console.progress_message(f"Building Docker image: {temp_tag}")

    # Build the image (env vars are for runtime, not build time)
    if not build_docker_image(env_dir, temp_tag, no_cache, verbose):
        hud_console.error("Docker build failed")
        raise typer.Exit(1)

    hud_console.success(f"Built temporary image: {temp_tag}")

    # Analyze the environment
    hud_console.progress_message("Analyzing MCP environment...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        analysis = loop.run_until_complete(analyze_mcp_environment(temp_tag, verbose, env_vars))
    finally:
        loop.close()

    hud_console.success(f"Analyzed environment: {analysis['toolCount']} tools found")

    # Extract environment variables from Dockerfile
    dockerfile_path = env_dir / "Dockerfile"
    required_env, optional_env = extract_env_vars_from_dockerfile(dockerfile_path)

    # Merge user-provided env vars with detected ones
    provided_env_vars = {}
    missing_required = []
    if env_vars:
        provided_env_vars = env_vars.copy()
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

    # Create lock file content - minimal and useful
    lock_content = {
        "version": "1.0",  # Lock file format version
        "image": tag,  # Will be updated with ID/digest later
        "build": {
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "hudVersion": hud_version,
            "directory": str(env_dir.name),
            "version": new_version,  # Internal environment version
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

    hud_console.success("Created lock file: hud.lock.yaml")

    # Calculate lock file hash
    lock_content_str = yaml.dump(lock_content, default_flow_style=False, sort_keys=True)
    lock_hash = hashlib.sha256(lock_content_str.encode()).hexdigest()
    lock_size = len(lock_content_str)

    # Rebuild with label containing lock file hash
    hud_console.progress_message("Rebuilding with lock file metadata...")

    # Build final image with label (uses cache from first build)
    # Also tag with version
    base_name = tag.split(":")[0] if tag and ":" in tag else tag
    version_tag = f"{base_name}:{new_version}"

    label_cmd = [
        "docker",
        "build",
        "--label",
        f"org.hud.manifest.head={lock_hash}:{lock_size}",
        "--label",
        f"org.hud.version={new_version}",
        "-t",
        tag,
        "-t",
        version_tag,
    ]

    label_cmd.append(str(env_dir))

    # Run rebuild using Docker's native output formatting
    if verbose:
        # Show Docker's native output when verbose
        result = subprocess.run(label_cmd, check=False)  # noqa: S603
    else:
        # Hide output when not verbose
        result = subprocess.run(  # noqa: S603
            label_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )

    if result.returncode != 0:
        hud_console.error("Failed to rebuild with label")
        raise typer.Exit(1)

    hud_console.success("Built final image with lock file metadata")

    # NOW get the image ID after the final build
    image_id = get_docker_image_id(tag)  # type: ignore
    if image_id:
        # For local builds, store the image ID
        # Docker IDs come as sha256:hash, we want tag@sha256:hash
        if image_id.startswith("sha256:"):
            lock_content["image"] = f"{tag}@{image_id}"
        else:
            lock_content["image"] = f"{tag}@sha256:{image_id}"

        # Update the lock file with the new image reference
        with open(lock_path, "w") as f:
            yaml.dump(lock_content, f, default_flow_style=False, sort_keys=False)

        hud_console.success("Updated lock file with image ID")
    else:
        hud_console.warning("Could not retrieve image ID for lock file")

    # Remove temp image after we're done
    subprocess.run(["docker", "rmi", temp_tag], capture_output=True)  # noqa: S603, S607

    # Add to local registry
    if image_id:
        # Save to local registry using the helper
        save_to_registry(lock_content, lock_content.get("image", tag), verbose)

    # Print summary
    hud_console.section_title("Build Complete")

    # Show the version tag as primary since that's what will be pushed
    hud_console.status_item("Built image", version_tag, primary=True)
    if tag:
        hud_console.status_item("Also tagged", tag)
    hud_console.status_item("Version", new_version)
    hud_console.status_item("Lock file", "hud.lock.yaml")
    hud_console.status_item("Tools found", str(analysis["toolCount"]))

    # Show the digest info separately if we have it
    if image_id:
        hud_console.dim_info("\nImage digest", image_id)

    hud_console.section_title("Next Steps")
    hud_console.info("Test locally:")
    hud_console.command_example("hud dev", "Hot-reload development")
    hud_console.command_example(f"hud run {tag}", "Run the built image")
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
) -> None:
    """Build a HUD environment and generate lock file."""
    build_environment(directory, tag, no_cache, verbose, env_vars)
