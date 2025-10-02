"""Fast metadata analysis functions for hud analyze."""

from __future__ import annotations

from urllib.parse import quote

import requests
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .registry import (
    extract_digest_from_image,
    list_registry_entries,
    load_from_registry,
)

console = Console()
hud_console = HUDConsole()


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

        return None
    except Exception:
        return None


def check_local_cache(reference: str) -> dict | None:
    """Check local cache for lock file."""
    # First try exact digest match
    digest = extract_digest_from_image(reference)
    lock_data = load_from_registry(digest)
    if lock_data:
        return lock_data

    # If not found and reference has a name, search by name pattern
    if "/" in reference:
        # Look for any cached version of this image
        ref_base = reference.split("@")[0].split(":")[0]

        for _, lock_file in list_registry_entries():
            try:
                with open(lock_file) as f:
                    lock_data = yaml.safe_load(f)
                # Check if this matches our reference
                if lock_data and "image" in lock_data:
                    image = lock_data["image"]
                    # Match by name (ignoring tag/digest)
                    img_base = image.split("@")[0].split(":")[0]
                    if ref_base in img_base or img_base in ref_base:
                        return lock_data
            except Exception:
                hud_console.error("Error loading lock file")

    return None


async def analyze_from_metadata(reference: str, output_format: str, verbose: bool) -> None:
    """Analyze environment from cached or registry metadata."""
    import json

    from hud.cli.analyze import display_interactive, display_markdown

    hud_console.header("MCP Environment Analysis", icon="🔍")
    hud_console.info(f"Looking up: {reference}")
    hud_console.info("")

    lock_data = None
    source = None

    # 1. Check local cache first
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking local cache...", total=None)

        lock_data = check_local_cache(reference)
        if lock_data:
            progress.update(task, description="[green]✓ Found in local cache[/green]")
            source = "local"
        else:
            progress.update(
                task, description="[yellow]→ Not in cache, checking registry...[/yellow]"
            )

            # 2. Try HUD registry
            # Parse reference to get org/name format
            if "/" in reference and "@" not in reference and ":" not in reference:
                # Already in org/name format
                registry_ref = reference
            elif "/" in reference:
                # Extract org/name from full reference
                parts = reference.split("/")
                if len(parts) >= 2:
                    # Handle docker.io/org/name or just org/name
                    if parts[0] in ["docker.io", "registry-1.docker.io", "index.docker.io"]:
                        # Remove registry prefix but keep tag
                        registry_ref = "/".join(parts[1:]).split("@")[0]
                    else:
                        # Keep org/name:tag format
                        registry_ref = "/".join(parts[:2]).split("@")[0]
                else:
                    registry_ref = reference
            else:
                registry_ref = reference

            if not settings.api_key:
                progress.update(
                    task, description="[yellow]→ No API key (checking public registry)...[/yellow]"
                )

            lock_data = fetch_lock_from_registry(registry_ref)
            if lock_data:
                progress.update(task, description="[green]✓ Found in HUD registry[/green]")
                source = "registry"

                # Save to local cache for next time
                from .registry import save_to_registry

                save_to_registry(lock_data, lock_data.get("image", ""), verbose=False)
            else:
                progress.update(task, description="[red]✗ Not found[/red]")

    if not lock_data:
        hud_console.error("Environment metadata not found")
        console.print("\n[yellow]This environment hasn't been analyzed yet.[/yellow]")
        console.print("\nOptions:")
        console.print(f"  1. Pull it first: [cyan]hud pull {reference}[/cyan]")
        console.print(f"  2. Run live analysis: [cyan]hud analyze {reference} --live[/cyan]")
        if not settings.api_key:
            console.print(
                "  3. Set HUD_API_KEY in your environment or run: hud set HUD_API_KEY=your-key-here"
            )
        return

    # Convert lock data to analysis format
    analysis = {
        "status": "metadata" if source == "local" else "registry",
        "source": source,
        "tools": [],
        "resources": [],
        "prompts": [],
    }

    # Add basic info
    if "image" in lock_data:
        analysis["image"] = lock_data["image"]

    if "build" in lock_data:
        analysis["build_info"] = lock_data["build"]

    if "push" in lock_data:
        analysis["push_info"] = lock_data["push"]

    # Extract environment info
    if "environment" in lock_data:
        env = lock_data["environment"]
        if "initializeMs" in env:
            analysis["init_time"] = env["initializeMs"]
        if "toolCount" in env:
            analysis["tool_count"] = env["toolCount"]
        if "variables" in env:
            analysis["env_vars"] = env["variables"]

    # Extract tools
    if "tools" in lock_data:
        for tool in lock_data["tools"]:
            analysis["tools"].append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}) if verbose else None,
                }
            )

    # Display results
    hud_console.info("")
    if source == "local":
        hud_console.dim_info("Source:", "Local cache")
    else:
        hud_console.dim_info("Source:", "HUD registry")

    if "image" in analysis:
        hud_console.dim_info("Image:", analysis["image"])

    hud_console.info("")

    # Display results based on format
    if output_format == "json":
        console.print_json(json.dumps(analysis, indent=2))
    elif output_format == "markdown":
        display_markdown(analysis)
    else:  # interactive
        display_interactive(analysis)
