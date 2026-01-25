"""Link local directory to existing HUD environment."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer

from hud.utils.hud_console import HUDConsole


def link_environment(
    directory: str = ".",
    registry_id: str | None = None,
    yes: bool = False,
) -> None:
    """Link a local directory to an existing HUD environment.

    Similar to 'vercel link' - creates .hud/deploy.json to connect
    this directory to an existing platform environment.

    Args:
        directory: Environment directory to link
        registry_id: Environment ID to link to (if not provided, will prompt)
        yes: Skip confirmation prompts
    """
    hud_console = HUDConsole()
    hud_console.header("HUD Environment Link")

    from hud.settings import settings

    env_dir = Path(directory).resolve()

    # Check for API key
    if not settings.api_key:
        hud_console.error("No HUD API key found")
        hud_console.info("Set your API key: hud set HUD_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Check if already linked
    deploy_link_path = env_dir / ".hud" / "deploy.json"
    if deploy_link_path.exists():
        try:
            with open(deploy_link_path) as f:
                existing_link = json.load(f)
            existing_id = existing_link.get("registryId")
            if existing_id:
                hud_console.warning(f"Already linked to: {existing_id[:8]}...")
                if not yes and not typer.confirm("Unlink and link to a different environment?"):
                    hud_console.info("Aborted.")
                    raise typer.Exit(0)
        except Exception:
            pass

    # If no registry_id provided, list available environments
    if not registry_id:
        hud_console.info("Fetching your environments...")

        try:
            response = httpx.get(
                f"{settings.hud_api_url.rstrip('/')}/registry/envs",
                headers={"X-API-Key": settings.api_key},
                params={"limit": 20, "sort_by": "updated_at"},
                timeout=30.0,
            )
            response.raise_for_status()
            envs = response.json()
        except httpx.HTTPStatusError as e:
            hud_console.error(f"Failed to fetch environments: {e.response.status_code}")
            raise typer.Exit(1) from e
        except Exception as e:
            hud_console.error(f"Failed to fetch environments: {e}")
            raise typer.Exit(1) from e

        if not envs:
            hud_console.warning("No environments found")
            hud_console.info("Deploy an environment first with: hud deploy")
            raise typer.Exit(1)

        # Display environments for selection
        hud_console.info("\nYour environments:")
        for i, env in enumerate(envs[:10], 1):
            env_id = env.get("id", "")[:8]
            env_name = env.get("name_display") or env.get("name", "unnamed")
            version = env.get("latest_version", "")
            version_str = f" v{version}" if version else ""
            hud_console.info(f"  {i}. {env_name}{version_str} ({env_id}...)")

        # Prompt for selection
        hud_console.info("")
        selection = typer.prompt(
            "Select environment number (or paste full ID)",
            default="1",
        )

        try:
            # Try as number first
            idx = int(selection) - 1
            if 0 <= idx < len(envs):
                registry_id = envs[idx]["id"]
            else:
                hud_console.error("Invalid selection")
                raise typer.Exit(1)
        except ValueError:
            # Assume it's a full ID
            registry_id = selection.strip()

    # Verify the environment exists and user has access
    hud_console.progress_message("Verifying environment...")

    try:
        response = httpx.get(
            f"{settings.hud_api_url.rstrip('/')}/registry/envs/{registry_id}",
            headers={"X-API-Key": settings.api_key},
            timeout=30.0,
        )
        response.raise_for_status()
        env_data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            hud_console.error(f"Environment not found: {registry_id}")
        else:
            hud_console.error(f"Failed to verify environment: {e.response.status_code}")
        raise typer.Exit(1) from e
    except Exception as e:
        hud_console.error(f"Failed to verify environment: {e}")
        raise typer.Exit(1) from e

    env_name = env_data.get("name_display") or env_data.get("name", "unnamed")
    version = env_data.get("latest_version", "")

    hud_console.success(f"Found: {env_name}")
    if version:
        hud_console.info(f"Latest version: {version}")

    # Confirm
    if not yes and not typer.confirm(f"\nLink {env_dir.name}/ to '{env_name}'?"):
        hud_console.info("Aborted.")
        raise typer.Exit(0)

    # Save link
    hud_dir = env_dir / ".hud"
    hud_dir.mkdir(parents=True, exist_ok=True)

    deploy_link = {
        "registryId": registry_id,
        "version": version,
    }

    try:
        with open(deploy_link_path, "w") as f:
            json.dump(deploy_link, f, indent=2)
        reg_id_short = registry_id[:8] if registry_id else "unknown"
        hud_console.success(f"Linked to: {env_name} ({reg_id_short}...)")
        hud_console.dim_info("Link stored in:", ".hud/deploy.json")
    except Exception as e:
        hud_console.error(f"Failed to save link: {e}")
        raise typer.Exit(1) from e

    # Show next steps
    hud_console.info("")
    hud_console.info("Next steps:")
    hud_console.info("  Deploy changes: hud deploy")
    hud_console.info("  Unlink: rm .hud/deploy.json")


def link_command(
    directory: str = typer.Argument(".", help="Directory to link"),
    registry_id: str | None = typer.Option(
        None,
        "--id",
        "-i",
        help="Environment ID to link to (prompts if not provided)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """🔗 Link directory to existing HUD environment.

    [not dim]Connects a local directory to an existing platform environment,
    so future 'hud deploy' commands update that environment.

    Similar to 'vercel link' for Vercel projects.

    Examples:
        hud link                    # Interactive selection
        hud link --id abc123...     # Link to specific environment
        hud link environments/browser[/not dim]
    """
    link_environment(
        directory=directory,
        registry_id=registry_id,
        yes=yes,
    )
