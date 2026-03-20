"""HUD RL command — submit validated tasks for RL training."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.table import Table

from hud.cli.utils.api import hud_headers, require_api_key
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole()


# =============================================================================
# Preflight validation
# =============================================================================


async def _fetch_env_metadata(env_name: str, headers: dict[str, str]) -> dict[str, Any] | None:
    """Fetch env metadata from mcp-config endpoint. Returns response dict or None."""
    url = f"{settings.hud_api_url}/environments/{env_name}/mcp-config"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            hud_console.error(f"Preflight check failed for '{env_name}': HTTP {resp.status_code}")
            raise typer.Exit(1)
        return resp.json()


def _check_scenarios(
    env_name: str,
    expected: set[str],
    env_data: dict[str, Any],
) -> None:
    """Check scenarios against platform data. Warns if surface unavailable."""
    scenarios = env_data.get("scenarios")
    if not isinstance(scenarios, list):
        hud_console.warning(f"Cannot verify scenarios for '{env_name}' (not exposed by platform)")
        return

    remote_names = set(scenarios)
    for scenario in sorted(expected):
        if scenario not in remote_names:
            hud_console.error(f"Scenario '{scenario}' not found on environment '{env_name}'")
            hud_console.hint(f"Available: {', '.join(sorted(remote_names))}")
            raise typer.Exit(1)
        display = scenario.removeprefix(f"{env_name}:")
        hud_console.info(f"  ✓ {env_name}:{display}")


def _extract_env_names(tasks: list[Any]) -> set[str]:
    """Extract unique environment names from tasks."""
    env_names: set[str] = set()
    for task in tasks:
        if hasattr(task, "env") and task.env is not None:
            env = task.env
            if hasattr(env, "name") and env.name:
                env_names.add(env.name)
        elif isinstance(task, dict):
            env = task.get("env")
            if isinstance(env, dict):
                name = env.get("name")
                if name:
                    env_names.add(name)
            elif isinstance(env, str):
                env_names.add(env)
    return env_names


def _extract_scenarios(tasks: list[Any]) -> dict[str, set[str]]:
    """Extract env_name -> {scenario_names} mapping from tasks."""
    mapping: dict[str, set[str]] = {}
    for task in tasks:
        env_name = None
        scenario = None
        if hasattr(task, "env") and task.env is not None:
            if hasattr(task.env, "name"):
                env_name = task.env.name
            scenario = getattr(task, "scenario", None)
        elif isinstance(task, dict):
            env = task.get("env")
            if isinstance(env, dict):
                env_name = env.get("name")
            elif isinstance(env, str):
                env_name = env
            scenario = task.get("scenario")

        if env_name and scenario:
            mapping.setdefault(env_name, set()).add(scenario)
    return mapping


async def _preflight_validate(tasks: list[Any]) -> None:
    """Pre-submission validation.

    Hard failures: missing env, missing API key, task load errors,
                   scenario mismatch (when scenario surface is available).
    Soft failures: scenario surface unavailable (warn + continue).
    """
    headers = hud_headers()
    env_names = _extract_env_names(tasks)

    if not env_names:
        hud_console.warning("No environment names found in tasks — skipping preflight")
        return

    hud_console.info(f"Preflight: checking {len(env_names)} environment(s)…")

    env_metadata: dict[str, dict[str, Any]] = {}
    for name in sorted(env_names):
        data = await _fetch_env_metadata(name, headers)
        if data is None:
            hud_console.error(f"Environment '{name}' not found on platform")
            hud_console.hint("Deploy it first with: hud deploy")
            raise typer.Exit(1)
        env_metadata[name] = data
        hud_console.info(f"  ✓ {name}")

    env_scenarios = _extract_scenarios(tasks)
    for env_name, scenarios in sorted(env_scenarios.items()):
        if env_name in env_metadata:
            _check_scenarios(env_name, scenarios, env_metadata[env_name])

    hud_console.success("Preflight passed")


# =============================================================================
# Model selection
# =============================================================================


def _fetch_models() -> list[dict[str, Any]]:
    """Fetch trainable models from the HUD API."""
    url = f"{settings.hud_api_url}/models/"
    headers = hud_headers()
    params = {"team_only": "true", "limit": 200}
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json().get("models", [])


def _select_model_interactive(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Display models and let user pick one."""
    import questionary

    trainable = [
        m
        for m in models
        if m.get("is_trainable", False)
        and m.get("status") == "ready"
        and not m.get("public", False)
        and m.get("model_name") is not None
    ]
    if not trainable:
        hud_console.error("No trainable models found in your team.")
        hud_console.hint("Fork a trainable model at https://hud.ai/models")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Provider")
    for i, m in enumerate(trainable, 1):
        provider = m.get("provider", {}).get("name", "unknown") if m.get("provider") else "unknown"
        table.add_row(str(i), m.get("name", "unnamed"), m.get("status", "unknown"), provider)
    hud_console.console.print(table)

    choices = [
        {"name": f"{m.get('name', 'unnamed')} ({m.get('base_model', 'unknown')})", "value": m}
        for m in trainable
    ]
    selected: dict[str, Any] = hud_console.select("Select a model to train:", choices)  # type: ignore[assignment]
    return selected


# =============================================================================
# Main command
# =============================================================================


def rl_run_command(
    source: str = typer.Argument(
        ...,
        help="Task source: local file (JSON/JSONL) or remote taskset name",
    ),
    model_id: str | None = typer.Option(
        None, "--model-id", "-m", help="Model ID to train (skip interactive selection)"
    ),
    reasoning_effort: str = typer.Option(
        "medium", "--reasoning-effort", help="Reasoning effort level (low, medium, high)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept all prompts"),
) -> None:
    """Submit tasks for RL training with preflight validation."""
    hud_console.header("HUD RL Training")

    require_api_key("submit RL training jobs")

    # Model selection (interactive — must happen before asyncio.run)
    selected_model_id: str
    if model_id:
        selected_model_id = model_id
        hud_console.info(f"Using model: {selected_model_id}")
    else:
        models = _fetch_models()
        if yes:
            trainable = [
                m
                for m in models
                if m.get("is_trainable", False)
                and m.get("status") == "ready"
                and not m.get("public", False)
                and m.get("model_name") is not None
            ]
            if not trainable:
                hud_console.error("No trainable models found.")
                raise typer.Exit(1)
            selected_model = trainable[0]
            hud_console.info(f"Auto-selected: {selected_model.get('name', 'unnamed')}")
        else:
            selected_model = _select_model_interactive(models)
        selected_model_id = selected_model["id"]
        hud_console.success(f"Model: {selected_model.get('name')} ({selected_model_id})")

    # Load tasks (sync)
    from hud.datasets.loader import _load_from_api, _load_from_file

    hud_console.info(f"Loading tasks from: {source}…")
    path = Path(source)
    try:
        if path.exists() and path.suffix in {".json", ".jsonl"}:
            tasks = _load_from_file(path)
        else:
            tasks, _taskset_id = _load_from_api(source)
    except Exception as e:
        hud_console.error(f"Failed to load tasks: {e}")
        raise typer.Exit(1) from e

    if not tasks:
        hud_console.error(f"No tasks found in: {source}")
        raise typer.Exit(1)

    hud_console.info(f"Loaded {len(tasks)} task(s)")

    # Preflight (async — env/scenario checks hit the platform API)
    asyncio.run(_preflight_validate(tasks))

    # Confirm
    hud_console.info(f"Tasks: {len(tasks)}")
    hud_console.info(f"Model: {selected_model_id}")
    hud_console.info(f"Reasoning effort: {reasoning_effort}")

    if not yes:
        import questionary

        if not questionary.confirm("Submit RL training job?", default=True).ask():
            hud_console.error("Cancelled")
            raise typer.Exit(0)

    # Serialize and submit (async)
    asyncio.run(_submit(tasks, selected_model_id, reasoning_effort))


async def _submit(
    tasks: list[Any],
    model_id: str,
    reasoning_effort: str,
) -> None:
    task_dicts: list[dict[str, Any]] = []
    for t in tasks:
        if hasattr(t, "model_dump"):
            task_dicts.append(t.model_dump(mode="json"))
        elif isinstance(t, dict):
            task_dicts.append(t)

    # Submits directly to RL service, not platform models API
    payload: dict[str, Any] = {
        "model_id": model_id,
        "dataset": {"tasks": task_dicts},
        "config": {"parameters": {"reasoning_effort": reasoning_effort}},
    }

    url = f"{settings.hud_rl_url}/training/jobs"
    headers = hud_headers({"Content-Type": "application/json"})

    hud_console.info("Submitting training job…")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            hud_console.error(f"Request failed ({resp.status_code}): {detail}")
            raise typer.Exit(1)

        data = resp.json()
        job_id = data.get("job_id")
        result_model_id = data.get("model", {}).get("id")

        hud_console.success(f"Training job submitted! ID: {job_id}")
        if result_model_id:
            hud_console.info(f"Model ID: {result_model_id}")
            hud_console.info(f"Check status: hud rft status {result_model_id}")

    except httpx.RequestError as e:
        hud_console.error(f"Connection error: {e}")
        raise typer.Exit(1) from e
