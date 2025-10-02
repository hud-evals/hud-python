"""Pull HUD environments from registry."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
import typer
import yaml
from rich.table import Table

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .utils.registry import save_to_registry


def get_docker_manifest(image: str) -> dict | None:
    """Get manifest from Docker registry without pulling the image."""
    try:
        # Try docker manifest inspect (requires experimental features)
        result = subprocess.run(  # noqa: S603
            ["docker", "manifest", "inspect", image],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            import json

            return json.loads(result.stdout)
    except Exception:
        return None


def get_image_size_from_manifest(manifest: dict) -> int | None:
    """Extract total image size from Docker manifest."""
    try:
        total_size = 0

        # Handle different manifest formats
        if "layers" in manifest:
            # v2 manifest
            for layer in manifest["layers"]:
                total_size += layer.get("size", 0)
        elif "manifests" in manifest:
            first_manifest = manifest["manifests"][0]
            total_size = first_manifest.get("size", 0)

        return total_size if total_size > 0 else None
    except Exception:
        return None


def fetch_lock_from_registry(reference: str) -> dict | None:
    """Fetch lock file from HUD registry."""
    try:
        # Reference should be org/name:tag format
        # If no tag specified, append :latest
        if "/" in reference and ":" not in reference:
            reference = f"{reference}:latest"

        # URL-encode the path segments to handle special characters in tags
        url_safe_path = "/".join(quote(part, safe="") for part in reference.split("/"))
        registry_url = f"{settings.hud_telemetry_url.rstrip('/')}/registry/envs/{url_safe_path}"

        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"

        response = requests.get(registry_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Parse the lock YAML from the response
            if "lock" in data:
                return yaml.safe_load(data["lock"])
            elif "lock_data" in data:
                return data["lock_data"]
            else:
                # Try to treat the whole response as lock data
                return data
        elif response.status_code == 404:
            # Not found - expected error, return None silently
            return None
        elif response.status_code == 401:
            # Authentication issue - might be a private environment
            return None
        else:
            # Other errors - also return None but could log if verbose
            return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def pull_environment(
    target: str,
    lock_file: str | None = None,
    yes: bool = False,
    verify_only: bool = False,
    verbose: bool = False,
) -> None:
    """Pull HUD environment from registry."""
    hud_console = HUDConsole()
    hud_console.header("HUD Environment Pull")

    # Two modes:
    # 1. Pull from lock file (recommended)
    # 2. Pull from image reference directly

    lock_data = None
    image_ref = target

    # Mode 1: Lock file provided
    if lock_file or target.endswith((".yaml", ".yml")):
        # If target looks like a lock file, use it
        if target.endswith((".yaml", ".yml")):
            lock_file = target

        lock_path = Path(lock_file) if lock_file else None
        if lock_path and not lock_path.exists():
            hud_console.error(f"Lock file not found: {lock_file}")
            raise typer.Exit(1)

        hud_console.info(f"Reading lock file: {lock_file}")
        if lock_path:
            with open(lock_path) as f:
                lock_data = yaml.safe_load(f)

        image_ref = lock_data.get("image", "") if lock_data else ""

    # Mode 2: Direct image reference
    else:
        # First, try to parse as org/env reference for HUD registry
        # Check if it's a simple org/name or org/name:tag format (no @sha256)
        if "/" in target and "@" not in target:
            # Looks like org/env reference, possibly with tag
            hud_console.info(f"Checking HUD registry for: {target}")

            # Check for API key (not required for pulling, but good to inform)
            if not settings.api_key:
                hud_console.info("No HUD API key set (pulling from public registry)")
                hud_console.info(
                    "Set it in your environment or run: hud set HUD_API_KEY=your-key-here"
                )

            lock_data = fetch_lock_from_registry(target)

            if lock_data:
                hud_console.success("Found in HUD registry")
                image_ref = lock_data.get("image", "")
            else:
                # Fall back to treating as Docker image
                if not settings.api_key:
                    hud_console.info(
                        "Not found in HUD registry (try setting HUD_API_KEY for private environments)"  # noqa: E501
                    )
                    hud_console.info(
                        "Set it in your environment or run: hud set HUD_API_KEY=your-key-here"
                    )
                else:
                    hud_console.info("Not found in HUD registry, treating as Docker image")

        # Try to get metadata from Docker registry
        if not lock_data:
            hud_console.info(f"Fetching Docker metadata for: {image_ref}")
            manifest = get_docker_manifest(image_ref)

            if manifest:
                # Create minimal lock data from manifest
                lock_data = {"image": image_ref, "source": "docker-manifest"}

                # Try to get size
                size = get_image_size_from_manifest(manifest)
                if size:
                    lock_data["size"] = format_size(size)

                if verbose:
                    hud_console.info(
                        f"Retrieved manifest (type: {manifest.get('mediaType', 'unknown')})"
                    )

    # Display environment summary
    hud_console.section_title("Environment Details")

    # Create summary table
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    # Image info - show simple name in table
    display_ref = image_ref.split("@")[0] if ":" in image_ref and "@" in image_ref else image_ref
    table.add_row("Image", display_ref)

    if lock_data:
        # Show size if available
        if "size" in lock_data:
            table.add_row("Size", lock_data["size"])

        # Check if this is full lock data or minimal manifest data
        if lock_data.get("source") == "docker-manifest":
            # Minimal data from Docker manifest
            table.add_row("Source", "Docker Registry")
            if not yes:
                hud_console.warning("Note: Limited metadata available from Docker registry.")
                hud_console.info("For full environment details, use a lock file.\n")
        else:
            # Full lock file data
            if "build" in lock_data:
                table.add_row("Built", lock_data["build"].get("generatedAt", "Unknown"))
                table.add_row("HUD Version", lock_data["build"].get("hudVersion", "Unknown"))

            if "environment" in lock_data:
                env_data = lock_data["environment"]
                table.add_row("Tools", str(env_data.get("toolCount", "Unknown")))
                table.add_row("Init Time", f"{env_data.get('initializeMs', 'Unknown')} ms")

            if "push" in lock_data:
                push_data = lock_data["push"]
                table.add_row("Registry", push_data.get("registry", "Unknown"))
                table.add_row("Pushed", push_data.get("pushedAt", "Unknown"))

            # Environment variables
            env_section = lock_data.get("environment", {})
            if "variables" in env_section:
                vars_data = env_section["variables"]
                if vars_data.get("required"):
                    table.add_row("Required Env", ", ".join(vars_data["required"]))
                if vars_data.get("optional"):
                    table.add_row("Optional Env", ", ".join(vars_data["optional"]))

    else:
        # No metadata available
        table.add_row("Source", "Unknown")

    # Use design's console to maintain consistent output
    hud_console.console.print(table)

    # Tool summary (show after table)
    if lock_data and "tools" in lock_data:
        hud_console.section_title("Available Tools")
        for tool in lock_data["tools"]:
            hud_console.info(f"• {tool['name']}: {tool['description']}")

    # Show warnings if no metadata
    if not lock_data and not yes:
        hud_console.warning("No metadata available for this image.")
        hud_console.info("The image will be pulled without verification.")

    # If verify only, stop here
    if verify_only:
        hud_console.success("Verification complete")
        return

    # Ask for confirmation unless --yes
    if not yes:
        hud_console.info("")
        # Show simple name for confirmation, not the full digest
        if ":" in image_ref and "@" in image_ref:
            simple_name = image_ref.split("@")[0]
        else:
            simple_name = image_ref
        if not typer.confirm(f"Pull {simple_name}?"):
            hud_console.info("Aborted")
            raise typer.Exit(0)

    # Pull the image
    hud_console.progress_message(f"Pulling {image_ref}...")

    # Run docker pull with progress
    process = subprocess.Popen(  # noqa: S603
        ["docker", "pull", image_ref],  # noqa: S607
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    for line in process.stdout or []:
        if verbose or "Downloading" in line or "Extracting" in line or "Pull complete" in line:
            hud_console.info(line.rstrip())

    process.wait()

    if process.returncode != 0:
        hud_console.error("Pull failed")
        raise typer.Exit(1)

    # Store lock file locally if we have full lock data (not minimal manifest data)
    if lock_data and lock_data.get("source") != "docker-manifest":
        # Save to local registry using the helper
        save_to_registry(lock_data, image_ref, verbose)

    # Success!
    hud_console.success("Pull complete!")

    # Show usage
    hud_console.section_title("Next Steps")

    # Extract simple name for examples
    simple_ref = image_ref.split("@")[0] if ":" in image_ref and "@" in image_ref else image_ref

    hud_console.info("1. Quick analysis (from metadata):")
    hud_console.command_example(f"hud analyze {simple_ref}")
    hud_console.info("")
    hud_console.info("2. Live analysis (runs container):")
    hud_console.command_example(f"hud analyze {simple_ref} --live")
    hud_console.info("")
    hud_console.info("3. Run the environment:")
    hud_console.command_example(f"hud run {simple_ref}")


def pull_command(
    target: str = typer.Argument(..., help="Image reference or lock file to pull"),
    lock_file: str | None = typer.Option(
        None, "--lock", "-l", help="Path to lock file (if target is image ref)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verify_only: bool = typer.Option(
        False, "--verify-only", help="Only verify metadata without pulling"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Pull HUD environment from registry with metadata preview."""
    pull_environment(target, lock_file, yes, verify_only, verbose)
