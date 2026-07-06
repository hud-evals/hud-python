"""``hud serve`` — serve a v6 :class:`~hud.environment.Environment` locally.

In v6, ``hud serve`` brings up an environment's control channel (tcp JSON-RPC)
so agents can connect to it. ``hud dev`` is a deprecated alias. The legacy
MCP-server hot-reload / Docker / inspector mode is no longer supported.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.markup import escape

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def _load_environment(module: str | None, factory_args: dict[str, str]) -> Any:
    """Resolve the serve target (``target[:name]``), printing failures."""
    from hud.environment import load_environment

    target, _, name = (module or "env").partition(":")
    try:
        return load_environment(target, name=name or None, args=factory_args or None)
    except Exception as exc:
        hud_console.error(str(exc))
        return None


def _serve_environment(env: Any, host: str, port: int) -> None:
    """Serve an ``Environment``'s control channel (tcp JSON-RPC) until interrupted."""
    hud_console.section_title("Environment")
    hud_console.console.print(
        f"{hud_console.sym.ITEM} {escape(env.name)}",
        highlight=False,
    )
    hud_console.console.print(
        f"{hud_console.sym.ITEM} serving on tcp://{host}:{port}",
        highlight=False,
    )
    hud_console.console.print(
        f"{hud_console.sym.ITEM} {len(env.tasks)} task(s), {len(env.capabilities)} capability(ies)",
        highlight=False,
    )
    hud_console.hint("Press Ctrl+C to stop.")
    from hud.environment.server import serve_blocking

    try:
        serve_blocking(env, host, port)
    except KeyboardInterrupt:
        hud_console.info("Stopped.")


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
    """🔥 Serve a HUD Environment locally (its tcp control channel).

    [not dim]Examples:
        hud serve                # auto-detect env.py
        hud serve env:env        # explicit module:attribute
        hud serve env.py -p 9000 # serve on a specific port
        hud serve pkg.mod:make_env --arg name=demo  # call a factory

    In v6, ``hud serve`` serves a :class:`hud.environment.Environment`. The old
    MCP-server hot-reload / Docker dev mode is no longer supported.[/not dim]
    """
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    factory_args: dict[str, str] = {}
    for pair in arg or []:
        key, _, value = pair.partition("=")
        factory_args[key] = value
    env = _load_environment(module, factory_args)
    if env is None:
        hud_console.error(
            f"No HUD Environment found for {module or 'env.py'}.",
        )
        hud_console.info(
            "In v6, `hud serve` serves a `hud.environment.Environment` "
            "(e.g. `env = Environment(name=...)` in env.py). "
            "MCP-server hot-reload mode is no longer supported.",
        )
        raise typer.Exit(1)

    _serve_environment(env, host, port)
