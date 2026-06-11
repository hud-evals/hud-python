"""``hud client`` — drive a running env's control channel from the shell.

A thin CLI over :class:`hud.clients.HudClient`. Point it at an env served by
``hud serve`` (or any control channel) to inspect it or run a task with a supplied
answer. The Harbor ``test.sh`` uses ``hud client run`` to grade.
"""

from __future__ import annotations

import asyncio
import json

import typer

from hud.eval.runtime import Runtime
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()

client_app = typer.Typer(
    help="Talk to a running env's control channel (served by `hud serve`).",
    rich_markup_mode="rich",
)


def _runtime(url: str) -> Runtime:
    return Runtime(url if "://" in url else f"tcp://{url}")


@client_app.command("info")
def info_command(
    url: str = typer.Option("tcp://127.0.0.1:8765", "--url", "-u", help="Env control-channel URL."),
) -> None:
    """Show the env's identity, capabilities, and tasks."""

    async def _run() -> None:
        from hud.clients import connect

        async with connect(_runtime(url), ready_timeout=10.0) as client:
            manifest = client.manifest
            if manifest is None:
                hud_console.error("No manifest returned by the env.")
                raise typer.Exit(1)
            hud_console.section_title("Environment")
            hud_console.info(f"{manifest.server_info.name} v{manifest.server_info.version}")
            hud_console.section_title("Capabilities")
            for cap in manifest.bindings:
                hud_console.info(f"  {cap.name}: {cap.protocol} -> {cap.url}")
            hud_console.section_title("Tasks")
            for task in await client.list_tasks():
                hud_console.info(f"  {task.get('id')}: {task.get('description', '')}")

    asyncio.run(_run())


@client_app.command("run")
def run_command(
    task: str = typer.Argument(..., help="Task id to run."),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    answer: str = typer.Option("", "--answer", help="Answer to submit as the result."),
    url: str = typer.Option("tcp://127.0.0.1:8765", "--url", "-u", help="Env control-channel URL."),
) -> None:
    """Start a task, submit an answer, and print the reward to stdout.

    Drives the control channel like an agent would, but the answer is supplied
    directly (e.g. by a Harbor ``test.sh`` via ``--answer "$(cat answer.txt)"``)
    instead of produced by an agent. The reward goes to stdout — redirect it where
    you need it (e.g. ``> /logs/verifier/reward.txt``).
    """

    async def _run() -> float:
        from hud.clients import connect
        from hud.eval.rollout import Run

        async with (
            connect(_runtime(url), ready_timeout=10.0) as client,
            Run(client, task, json.loads(args)) as run,
        ):
            run.trace.content = answer
        return run.reward

    typer.echo(str(asyncio.run(_run())))
