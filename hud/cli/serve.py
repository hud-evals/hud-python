"""``hud serve`` — serve a v6 :class:`~hud.environment.Environment` locally.

In v6, ``hud serve`` brings up an environment's control channel (tcp JSON-RPC)
so agents can connect to it.
"""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.markup import escape

from hud.cli import parse_key_value
from hud.environment import load_environment
from hud.environment.server import serve
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def serve_command(
    module: str | None = typer.Argument(
        None,
        help="An env source ('env:env', 'env', 'env.py') or a factory "
        "('pkg.mod:make_env' with --arg).",
    ),
    port: int = typer.Option(
        8765, "--port", "-p", help="Port to serve the environment control channel on."
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Interface to bind (use 0.0.0.0 inside containers)."
    ),
    arg: list[str] | None = typer.Option(  # noqa: B008
        None, "--arg", help="key=value passed to a factory target (repeatable)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs."),
) -> None:
    """Serve a HUD Environment locally (its tcp control channel).

    [not dim]Examples:
        hud serve                # auto-detect env.py
        hud serve env:env        # explicit module:attribute
        hud serve env.py -p 9000 # serve on a specific port
        hud serve pkg.mod:make_env --arg name=demo  # call a factory

    In v6, ``hud serve`` serves a :class:`hud.environment.Environment`. The old
    MCP-server hot-reload / Docker dev mode is no longer supported.[/not dim]
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    factory_args: dict[str, str] = {}
    for pair in arg or []:
        parsed = parse_key_value(pair)
        if parsed is None:
            raise ValueError(f"--arg expects key=value, got {pair!r}")
        factory_args[parsed[0]] = parsed[1]
    target, _, name = (module or "env").partition(":")
    env = load_environment(target, name=name or None, args=factory_args or None)

    hud_console.section_title("Environment")
    hud_console.print(f"{hud_console.sym.ITEM} {escape(env.name)}", highlight=False)
    hud_console.print(f"{hud_console.sym.ITEM} serving on tcp://{host}:{port}", highlight=False)
    hud_console.print(
        f"{hud_console.sym.ITEM} {len(env.tasks)} task(s), {len(env.capabilities)} capability(ies)",
        highlight=False,
    )
    hud_console.hint("Press Ctrl+C to stop.")
    try:
        asyncio.run(serve(env, host, port))
    except KeyboardInterrupt:
        hud_console.info("Stopped.")
