"""Run an A2A orchestrator that auto-discovers environment scenarios.

Usage:
    HUD_ENV=my-hud-environment HUD_MODEL=claude-haiku-4-5 \
      uv run python examples/03_a2a_environment_orchestrator.py
"""

from __future__ import annotations

import asyncio
import os

from hud.services import OrchestratorExecutor


def main() -> None:
    env_name = os.getenv("HUD_ENV", "").strip()
    if not env_name:
        raise ValueError("Set HUD_ENV to the target environment name.")

    model = os.getenv("HUD_MODEL", "claude-haiku-4-5")
    main_model = os.getenv("HUD_MAIN_MODEL", "gpt-4o")
    host = os.getenv("HUD_A2A_HOST", "0.0.0.0")
    port = int(os.getenv("HUD_A2A_PORT", "9999"))

    orchestrator = asyncio.run(
        OrchestratorExecutor.from_environment(
            env_name,
            model=model,
            main_model=main_model,
        )
    )
    orchestrator.serve(host=host, port=port)


if __name__ == "__main__":
    main()
