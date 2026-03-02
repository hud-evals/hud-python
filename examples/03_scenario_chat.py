"""Interactive REPL for scenario chat with optional streaming.

Usage:
    HUD_API_KEY=... HUD_ENV_NAME=... python examples/03_interactive_repl.py
    HUD_API_KEY=... HUD_ENV_NAME=... python examples/03_interactive_repl.py --stream
"""

from __future__ import annotations

import argparse
import asyncio
import os

import hud
from openai import AsyncOpenAI

TURN_TIMEOUT_SECONDS = 60


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true", help="Enable SSE token streaming.")
    parser.add_argument(
        "--env",
        default=os.getenv("HUD_ENV_NAME"),
        help="HUD environment name (or set HUD_ENV_NAME).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("HUD_MODEL", "gpt-4o"),
        help="Model name for chat calls.",
    )
    return parser


async def main() -> None:
    args = _parser().parse_args()
    if not args.env:
        raise ValueError("Provide --env or set HUD_ENV_NAME")

    client = AsyncOpenAI(
        base_url="https://inference.hud.ai",
        api_key=os.environ["HUD_API_KEY"],
    )
    env = hud.Environment(args.env)
    env.connect_hub(args.env)

    async with env:
        scenarios = await env.list_scenarios()
        if not scenarios:
            print("No scenarios found.")
            return

        print("Available scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            req = ", ".join(scenario.required_args) or "(none)"
            print(f"  [{i}] {scenario.short_name} - {scenario.description or 'no description'}")
            print(f"      required args: {req}")
        print()

        choice = input("Pick a scenario (number, default 1): ").strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        chosen = scenarios[idx] if 0 <= idx < len(scenarios) else scenarios[0]

        scenario_args: dict[str, str] = {}
        for arg in chosen.arguments:
            label = arg.name if arg.required else f"{arg.name} (optional)"
            value = input(f"  {label}: ").strip()
            if value:
                scenario_args[arg.name] = value

        print(f"\nRunning: {chosen.short_name}")
        print(f"Streaming: {'on' if args.stream else 'off'}")
        print("Type /done when finished.\n")

        async with hud.run_scenario_chat_interactive(
            client=client,
            model=args.model,
            env=env,
            scenario=chosen.short_name,
            args=scenario_args,
            api="chat_completions",
        ) as chat:
            print(f"Trace: https://hud.ai/trace/{chat.trace_id}\n")

            async def send_message(msg: str) -> None:
                if args.stream:
                    print("Assistant: ", end="", flush=True)
                    async for event in chat.send_stream(msg):
                        if event.type == "text_delta":
                            print(event.content, end="", flush=True)
                    print("\n")
                    return
                turn = await asyncio.wait_for(chat.send(msg), timeout=TURN_TIMEOUT_SECONDS)
                print(f"Assistant: {turn.answer}\n")

            await send_message("Begin.")
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_input:
                    continue
                if user_input.lower() in {"/done", "/quit", "/exit"}:
                    break
                await send_message(user_input)

            result = await chat.finish()
            print("---")
            print(f"Reward: {result.reward}")
            print(f"Trace:  https://hud.ai/trace/{result.trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
