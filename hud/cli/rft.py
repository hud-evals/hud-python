from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from hud.settings import settings
from hud.utils.hud_console import HUDConsole
from hud.utils.tasks import load_tasks

logger = logging.getLogger(__name__)
console = Console()

def rft_command(
    tasks_file: str,
    provider: str = "openai",
    reasoning_effort: str = "medium",
    verbose: bool = False,
) -> None:
    """
    Run Reinforcement Fine-Tuning (RFT) via the HUD RL service.
    """
    hud_console = HUDConsole()
    
    if not settings.api_key:
        hud_console.error("HUD_API_KEY not found in environment.")
        hud_console.info("Run 'hud set HUD_API_KEY=...' or export it.")
        raise typer.Exit(1)

    # Load tasks just to validate file existence and maybe show count
    try:
        tasks = load_tasks(tasks_file, raw=True) # Load raw dicts
        if not tasks:
            hud_console.error(f"No tasks found in {tasks_file}")
            raise typer.Exit(1)
        hud_console.info(f"Loaded {len(tasks)} tasks from {tasks_file}")
    except Exception as e:
        hud_console.error(f"Failed to load tasks file: {e}")
        raise typer.Exit(1)

    # Prepare payload
    payload = {
        "provider": provider,
        "base_model": "o4-mini-2025-04-16",
        "dataset": {
            "tasks": tasks
        },
        "config": {
            "parameters": {
                "reasoning_effort": reasoning_effort
            }
        }
    }

    # Send request to service
    import httpx
    
    base_url = settings.hud_rl_url


    url = f"{base_url}/v1/training/jobs"
    
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json"
    }

    hud_console.info(f"Submitting job to {url}...")
    
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except:
                    detail = resp.text
                hud_console.error(f"Request failed ({resp.status_code}): {detail}")
                raise typer.Exit(1)
            
            data = resp.json()
            job_id = data.get("job_id")
            model_id = data.get("model").get("id")
            
            hud_console.success(f"Job launched successfully! ID: {job_id}")
            hud_console.info(f"Model ID: {model_id}")

    except httpx.RequestError as e:
        hud_console.error(f"Connection error: {e}")
        hud_console.info("Is the RL service running?")
        raise typer.Exit(1)
