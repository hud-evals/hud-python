from __future__ import annotations

import logging
from typing import Any

import typer
from rich.console import Console

from hud.settings import settings
from hud.utils.hud_console import HUDConsole
from hud.utils.tasks import load_tasks

logger = logging.getLogger(__name__)
console = Console()
hud_console = HUDConsole()


def _patch_mcp_urls_to_staging(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively patch all mcp.hud.so URLs to https://orcstaging.hud.so in task configs."""

    def patch_value(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: patch_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [patch_value(item) for item in obj]
        elif isinstance(obj, str):
            # Replace any occurrence of mcp.hud.so with orcstaging.hud.so
            # Handle various URL formats
            if "mcp.hud.so" in obj:
                # Replace the domain while preserving the protocol and path
                return obj.replace("mcp.hud.so", "orcstaging.hud.so")
            elif "mcp.hud.ai" in obj:
                # Also handle mcp.hud.ai URLs
                return obj.replace("mcp.hud.ai", "orcstaging.hud.so")
            return obj
        else:
            return obj

    return [patch_value(task) for task in tasks]


def rft_command(
    tasks_file: str,
    provider: str = "openai",
    reasoning_effort: str = "medium",
    verbose: bool = False,
    yes: bool = False,
) -> None:
    """
    Run Reinforcement Fine-Tuning (RFT) via the HUD RL service.
    """
    hud_console.header("HUD RFT (Reinforcement Fine-Tuning)")

    # Preflight check: API key
    if not settings.api_key:
        hud_console.error("HUD_API_KEY not found in environment.")
        hud_console.info("Run 'hud set HUD_API_KEY=...' or export it.")
        raise typer.Exit(1)

    # Preflight check: Convert tasks to remote if needed
    hud_console.section_title("Preparing tasks for remote training")
    try:
        from hud.cli.flows.tasks import convert_tasks_to_remote

        hud_console.info("Checking task configuration...")
        tasks_file = convert_tasks_to_remote(tasks_file)
        hud_console.success("Tasks are ready for remote training")
    except typer.Exit:
        raise
    except Exception as e:
        hud_console.error(f"Tasks file is not valid for remote training: {e!s}")
        hud_console.hint("Either ensure the tasks file has remote urls")
        hud_console.hint("Or run 'hud rft' within an environment directory")
        raise typer.Exit(1) from e

    # Load and validate tasks
    try:
        # Load tasks with env vars already resolved
        from hud.types import Task  # noqa: TC001

        tasks_objects: list[Task] = load_tasks(tasks_file)  # type: ignore[assignment]
        # Convert to dicts for patching and serialization
        tasks: list[dict[str, Any]] = [t.model_dump() for t in tasks_objects]
        if not tasks:
            hud_console.error(f"No tasks found in {tasks_file}")
            raise typer.Exit(1)

        # Preflight check: Minimum task count
        task_count = len(tasks)
        if task_count < 10:
            hud_console.error(
                f"Insufficient tasks for RFT training: found {task_count}, need at least 10"
            )
            hud_console.hint("RFT requires a minimum of 10 tasks for effective training")
            raise typer.Exit(1)

        hud_console.info(f"Loaded {task_count} tasks from {tasks_file}")

        # Preflight check: Vision support
        hud_console.section_title("Vision Support Check")
        hud_console.warning(
            "RFT does not currently support environments that require vision capabilities."
        )
        hud_console.info(
            "Vision support includes: screenshots, image analysis, visual UI interaction, etc."
        )

        if not yes:
            if hud_console.confirm("Does your environment require vision support?", default=False):
                hud_console.error("RFT does not support vision-based environments at this time.")
                hud_console.hint(
                    "Please use environments that rely on text-based interactions only."
                )
                raise typer.Exit(1)
        else:
            hud_console.info("Skipping vision support check (--yes mode)")

        # Patch all mcp.hud.so URLs to orcstaging.hud.so
        hud_console.info("Patching MCP URLs for staging environment...")
        tasks = _patch_mcp_urls_to_staging(tasks)

        # Show task preview
        if tasks:
            if yes:
                # Skip interactive preview in auto-accept mode
                hud_console.info("Skipping task preview in auto-accept mode (--yes)")
            else:
                try:
                    from hud.cli.rl.viewer import show_json_interactive

                    hud_console.section_title("Task Preview")
                    show_json_interactive(
                        tasks[0], title="Example Task from Dataset", initial_expanded=False
                    )
                    hud_console.info("This is how your task will be sent to the RFT service.")

                    # Ask for confirmation
                    if not hud_console.confirm(
                        "\nProceed with RFT training on this dataset?", default=True
                    ):
                        hud_console.error("RFT training cancelled")
                        raise typer.Exit(0)
                except typer.Exit:
                    raise  # Re-raise typer.Exit to properly exit on cancellation
                except Exception as e:
                    hud_console.warning(f"Could not display task preview: {e}")

    except typer.Exit:
        raise  # Re-raise typer.Exit to properly exit
    except Exception as e:
        hud_console.error(f"Failed to load tasks file: {e}")
        raise typer.Exit(1) from e

    # Prepare payload
    payload = {
        "provider": provider,
        "base_model": "o4-mini-2025-04-16",
        "dataset": {"tasks": tasks},
        "config": {"parameters": {"reasoning_effort": reasoning_effort}},
    }

    # Send request to service
    hud_console.section_title("Submitting RFT job")

    import httpx

    base_url = settings.hud_rl_url
    url = f"{base_url}/training/jobs"

    headers = {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}

    hud_console.info(
        f"Submitting job to {url}... (this may take a few minutes to run all safety checks)"
    )

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload, headers=headers)

            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception as e:
                    detail = f"{resp.text} - {e}"
                hud_console.error(f"Request failed ({resp.status_code}): {detail}")
                raise typer.Exit(1)

            data = resp.json()
            job_id = data.get("job_id")
            model_id = data.get("model", {}).get("id")

            hud_console.success(f"Job launched successfully! ID: {job_id}")
            hud_console.info(f"Model ID: {model_id}")

            # Provide helpful next steps
            hud_console.info(f"To check job status, run: hud rft status {model_id}")

    except httpx.RequestError as e:
        hud_console.error(f"Connection error: {e}")
        hud_console.info("Is the RL service running?")
        raise typer.Exit(1) from e
