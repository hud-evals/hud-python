"""Serve HUD scenarios as an OpenAI-compatible agent endpoint.

Usage:
    HUD_API_KEY=... HUD_ENV_NAME=... python examples/04_scenario_server.py
"""

from __future__ import annotations

import os

import hud
from openai import AsyncOpenAI


def main() -> None:
    env_name = os.getenv("HUD_ENV_NAME")
    if not env_name:
        raise ValueError("Set HUD_ENV_NAME to the target HUD environment")

    model = os.getenv("HUD_MODEL", "gpt-4o")
    port = int(os.getenv("HUD_AGENT_PORT", "8321"))

    client = AsyncOpenAI(
        base_url="https://inference.hud.ai",
        api_key=os.environ["HUD_API_KEY"],
    )
    env = hud.Environment(env_name)
    env.connect_hub(env_name)

    print(f"Serving {env_name} on http://localhost:{port}")
    for route in (
        "GET  /scenarios",
        "POST /v1/chat/completions (use X-HUD-Session-Id for follow-up turns)",
        "POST /v1/sessions/<id>/finish",
        "GET  /v1/sessions",
    ):
        print(route)
    print()

    env.serve_as_agent(client=client, model=model, port=port)


if __name__ == "__main__":
    main()
