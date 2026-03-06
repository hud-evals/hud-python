"""Run an A2A server that forwards messages to a HUD environment.

The environment defines its own tools, system prompt, and routing via a
``chat=True`` scenario.  This script just wraps it with A2A session
management and serves it.

Usage:
    # Multi-scenario router mode
    HUD_ENV=my-hud-environment HUD_SCENARIO=assist \
        uv run python examples/03_a2a_environment_orchestrator.py

    # Single-scenario direct mode
    HUD_ENV=my-hud-environment HUD_SCENARIO=analysis_chat \
        uv run python examples/03_a2a_environment_orchestrator.py
"""

from __future__ import annotations

import os

from hud.services import OrchestratorExecutor


def main() -> None:
    env_name = os.getenv("HUD_ENV", "").strip()
    if not env_name:
        raise ValueError("Set HUD_ENV to the target environment name.")

    model = os.getenv("HUD_MODEL", "claude-haiku-4-5")
    scenario = os.getenv("HUD_SCENARIO", "assist").strip() or "assist"
    host = os.getenv("HUD_A2A_HOST", "0.0.0.0")
    port = int(os.getenv("HUD_A2A_PORT", "9999"))

    orchestrator = OrchestratorExecutor(env_name, model=model, scenario=scenario)
    orchestrator.serve(host=host, port=port)


if __name__ == "__main__":
    main()
