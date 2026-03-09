"""Run an A2A server that forwards messages to a HUD environment.

The environment defines its own tools, system prompt, and routing via a
``chat=True`` scenario.  This script just wraps it with A2A session
management and serves it.

Usage:
    HUD_ENV=my-hud-environment HUD_SCENARIO=analysis_chat \
        uv run python examples/03_a2a_chat_server.py

The configured scenario should be ``chat=True`` and accept a ``messages``
argument for multi-turn history.
"""

from __future__ import annotations

import os

from hud.eval.task import Task
from hud.services import ChatService


def main() -> None:
    env_name = os.getenv("HUD_ENV", "").strip()
    if not env_name:
        raise ValueError("Set HUD_ENV to the target environment name.")

    model = os.getenv("HUD_MODEL", "claude-haiku-4-5")
    scenario = os.getenv("HUD_SCENARIO", "").strip()
    if not scenario:
        raise ValueError("Set HUD_SCENARIO to the target chat scenario name.")
    host = os.getenv("HUD_A2A_HOST", "0.0.0.0")
    port = int(os.getenv("HUD_A2A_PORT", "9999"))

    resolved_scenario = scenario if ":" in scenario else f"{env_name}:{scenario}"
    service = ChatService(
        Task(env={"name": env_name}, scenario=resolved_scenario),
        model=model,
    )
    service.serve(host=host, port=port)


if __name__ == "__main__":
    main()
