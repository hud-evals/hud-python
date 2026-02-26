"""Fast metadata analysis functions for hud analyze."""

from __future__ import annotations

from urllib.parse import quote

import requests
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from hud.settings import settings
from hud.utils.hud_console import HUDConsole

from .api import hud_headers

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
        registry_url = f"{settings.hud_api_url.rstrip('/')}/registry/envs/{url_safe_path}"

        headers = hud_headers()

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


async def analyze_from_metadata(reference: str, output_format: str, verbose: bool) -> None:
    """Analyze environment from cached or registry metadata."""
    import json

    from hud.cli.analyze import display_interactive, display_markdown

    hud_console.header("MCP Environment Analysis", icon="🔍")
    hud_console.info(f"Looking up: {reference}")
    hud_console.info("")

    lock_data = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking HUD registry...", total=None)

        # Parse reference to get org/name format
        if "/" in reference and "@" not in reference and ":" not in reference:
            registry_ref = reference
        elif "/" in reference:
            parts = reference.split("/")
            if len(parts) >= 2:
                if parts[0] in ["docker.io", "registry-1.docker.io", "index.docker.io"]:
                    registry_ref = "/".join(parts[1:]).split("@")[0]
                else:
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
        else:
            progress.update(task, description="[red]✗ Not found[/red]")

    if not lock_data:
        hud_console.error("Environment metadata not found")
        console.print("\n[yellow]This environment hasn't been analyzed yet.[/yellow]")
        console.print("\nOptions:")
        console.print(f"  1. Run live analysis: [cyan]hud analyze {reference} --live[/cyan]")
        if not settings.api_key:
            console.print(
                "  3. Set HUD_API_KEY in your environment or run: hud set HUD_API_KEY=your-key-here"
            )
        return

    # Convert lock data to analysis format
    analysis = {
        "status": "registry",
        "source": "registry",
        "tools": [],
        "resources": [],
        "prompts": [],
        "scenarios": [],
        "verbose": verbose,
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

    # Extract resources
    if "resources" in lock_data:
        for resource in lock_data["resources"]:
            analysis["resources"].append(
                {
                    "uri": resource.get("uri", ""),
                    "name": resource.get("name", ""),
                    "description": resource.get("description", ""),
                    "mime_type": resource.get("mimeType", resource.get("mime_type", "")),
                }
            )

    # Extract prompts
    if "prompts" in lock_data:
        for prompt in lock_data["prompts"]:
            analysis["prompts"].append(
                {
                    "name": prompt.get("name", ""),
                    "description": prompt.get("description", ""),
                    "arguments": prompt.get("arguments", []),
                }
            )

    # Derive scenarios from scenario prompts/resources if present
    scenarios_by_id: dict[str, dict] = {}
    for p in analysis["prompts"]:
        desc = (p.get("description") or "").strip()
        if not desc.startswith("[Setup]"):
            continue
        scenario_id = p.get("name")
        if not scenario_id:
            continue
        env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
        scenarios_by_id[scenario_id] = {
            "id": scenario_id,
            "env": env_name,
            "name": scenario_name or scenario_id,
            "setup_description": desc,
            "arguments": p.get("arguments") or [],
            "has_setup_prompt": True,
            "has_evaluate_resource": False,
        }
    for r in analysis["resources"]:
        desc = (r.get("description") or "").strip()
        if not desc.startswith("[Evaluate]"):
            continue
        scenario_id = r.get("uri")
        if not scenario_id:
            continue
        env_name, scenario_name = ([*scenario_id.split(":", 1), ""])[:2]
        if scenario_id not in scenarios_by_id:
            scenarios_by_id[scenario_id] = {
                "id": scenario_id,
                "env": env_name,
                "name": scenario_name or scenario_id,
                "arguments": [],
                "has_setup_prompt": False,
                "has_evaluate_resource": True,
            }
        scenarios_by_id[scenario_id]["evaluate_description"] = desc
        scenarios_by_id[scenario_id]["has_evaluate_resource"] = True

    analysis["scenarios"] = sorted(
        scenarios_by_id.values(),
        key=lambda s: (str(s.get("env") or ""), str(s.get("name") or "")),
    )

    # Display results
    hud_console.info("")
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
