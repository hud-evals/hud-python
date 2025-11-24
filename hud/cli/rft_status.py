from __future__ import annotations

import logging

import httpx
import typer
from rich.console import Console

from hud.cli.rl.viewer import show_json_interactive
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
console = Console()
hud_console = HUDConsole()


def rft_status_command(
    model_id: str,
    verbose: bool = False,
) -> None:
    """
    Check the status of an RFT training job.
    """
    # hud_console.header("RFT Job Status")

    # Preflight check: API key
    if not settings.api_key:
        hud_console.error("HUD_API_KEY not found in environment.")
        hud_console.info("Run 'hud set HUD_API_KEY=...' or export it.")
        raise typer.Exit(1)

    # Prepare request
    base_url = settings.hud_rl_url
    url = f"{base_url}/training/jobs/{model_id}/raw-status"

    headers = {"Authorization": f"Bearer {settings.api_key}"}

    hud_console.info(f"Fetching status for model: {model_id}")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)

            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception as e:
                    detail = f"{resp.text} - {e}"
                hud_console.error(f"Request failed ({resp.status_code}): {detail}")
                raise typer.Exit(1)

            data = resp.json()

            # Display status information
            status = data.get("status", "Unknown")

            # Show status with appropriate styling
            if status.lower() in ["succeeded", "completed"]:
                hud_console.success(f"Job Status: {status}")
            elif status.lower() in ["failed", "error", "cancelled"]:
                hud_console.error(f"Job Status: {status}")
            elif status.lower() in [
                "running",
                "in_progress",
                "processing",
                "validating_files",
                "queued",
            ]:
                hud_console.info(f"Job Status: {status} 🔄")
            else:
                hud_console.info(f"Job Status: {status}")

            # Most important: Show fine-tuned model if available
            if data.get("fine_tuned_model"):
                hud_console.success(f"Fine-tuned Model: {data['fine_tuned_model']}")
                console.print("\n[dim]You can now use this model in your applications![/dim]")

            # Display full response in verbose mode or interactive viewer
            if verbose:
                hud_console.section_title("Full Status Details")
                show_json_interactive(data, title="RFT Job Status", initial_expanded=True)
            else:
                # Show key information
                if "model" in data:
                    hud_console.info(f"Base Model: {data['model']}")

                if "created_at" in data:
                    # Convert timestamp to readable format if it's a unix timestamp
                    created = data["created_at"]
                    if isinstance(created, int) and created > 1000000000:
                        from datetime import datetime

                        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
                        hud_console.info(f"Created: {created_str}")
                    else:
                        hud_console.info(f"Created: {created}")

                if data.get("finished_at"):
                    finished = data["finished_at"]
                    if isinstance(finished, int) and finished > 1000000000:
                        from datetime import datetime

                        finished_str = datetime.fromtimestamp(finished).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        hud_console.info(f"Finished: {finished_str}")
                    else:
                        hud_console.info(f"Finished: {finished}")

                if data.get("trained_tokens"):
                    hud_console.info(f"Trained Tokens: {data['trained_tokens']:,}")

                if (
                    "estimated_finish" in data
                    and data["estimated_finish"]
                    and data["estimated_finish"] > 0
                ):
                    from datetime import datetime

                    est_str = datetime.fromtimestamp(data["estimated_finish"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    hud_console.info(f"Estimated Finish: {est_str}")

                # Only show error if it's actually an error (not empty/null)
                if data.get("error"):
                    error = data["error"]
                    # Check if it's a real error
                    if isinstance(error, dict):
                        # Check if any field has actual content
                        has_content = any(error.get(k) for k in ["code", "message", "param"])
                        if has_content:
                            error_msg = error.get("message") or str(error)
                            hud_console.error(f"Error: {error_msg}")
                    elif isinstance(error, str) and error.strip():
                        hud_console.error(f"Error: {error}")

                # Suggest verbose mode for more details
                console.print("\n[dim]Use --verbose to see full status details[/dim]")

    except httpx.RequestError as e:
        hud_console.error(f"Connection error: {e}")
        hud_console.info("Is the RL service running?")
        raise typer.Exit(1) from e
