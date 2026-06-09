"""``hud dev`` — serve a v6 :class:`~hud.environment.Environment` locally.

In v6, ``hud dev`` brings up an environment's control channel (tcp JSON-RPC) so
agents can connect to it. The legacy MCP-server hot-reload / Docker / inspector
mode is no longer supported.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def _load_environment(module: str | None) -> Any:
    """Load a v6 :class:`~hud.environment.Environment` from a dev target.

    Accepts ``None`` (defaults to ``env.py``), ``module``, ``module:attr``, or a
    ``path/to/env.py``. Returns the ``Environment`` instance, or ``None`` if the
    target isn't a v6 environment.
    """
    from hud.environment import Environment
    from hud.eval import load_module

    target, _, attr = (module or "env").partition(":")
    path = Path(target)
    if path.suffix != ".py":
        path = Path(f"{target}.py")
    if not path.exists():
        return None
    try:
        mod = load_module(path)
    except Exception as exc:
        hud_console.error(f"Failed to import {path}: {exc}")
        return None
    if attr:
        obj = getattr(mod, attr, None)
        return obj if isinstance(obj, Environment) else None
    envs = [v for v in vars(mod).values() if isinstance(v, Environment)]
    if len(envs) > 1:
        hud_console.error(
            f"Multiple Environments found in {path}; specify one with 'module:attr'.",
        )
        return None
    return envs[0] if envs else None


def _serve_environment(env: Any, port: int) -> None:
    """Serve an ``Environment``'s control channel (tcp JSON-RPC) until interrupted."""
    hud_console.section_title("Environment")
    hud_console.console.print(
        f"{hud_console.sym.ITEM} {escape(env.name)}",
        highlight=False,
    )
    hud_console.console.print(
        f"{hud_console.sym.ITEM} serving on tcp://127.0.0.1:{port}",
        highlight=False,
    )
    hud_console.console.print(
        f"{hud_console.sym.ITEM} {len(env.task_entries())} task(s), "
        f"{len(env.capabilities)} capability(ies)",
        highlight=False,
    )
    hud_console.hint("Press Ctrl+C to stop.")
    try:
        asyncio.run(env.serve("127.0.0.1", port))
    except KeyboardInterrupt:
        hud_console.info("Stopped.")


def dev_command(
    module: str | None = typer.Argument(
        None,
        help="Module exposing an Environment (e.g. 'env:env', 'env', or 'env.py').",
    ),
    port: int = typer.Option(
        8765, "--port", "-p", help="Port to serve the environment control channel on."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs."),
) -> None:
    """🔥 Serve a HUD Environment locally (its tcp control channel).

    [not dim]Examples:
        hud dev                # auto-detect env.py
        hud dev env:env        # explicit module:attribute
        hud dev env.py -p 9000 # serve on a specific port

    In v6, ``hud dev`` serves a :class:`hud.environment.Environment`. The old
    MCP-server hot-reload / Docker dev mode is no longer supported.[/not dim]
    """
    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    env = _load_environment(module)
    if env is None:
        hud_console.error(
            f"No HUD Environment found for {module or 'env.py'}.",
        )
        hud_console.info(
            "In v6, `hud dev` serves a `hud.environment.Environment` "
            "(e.g. `env = Environment(name=...)` in env.py). "
            "MCP-server hot-reload mode is no longer supported.",
        )
        raise typer.Exit(1)

    _serve_environment(env, port)
