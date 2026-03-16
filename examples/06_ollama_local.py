"""Ollama local model demo — run evals on Llama, Mistral, etc. locally.

Uses OllamaAgent to connect to a local Ollama server. No API keys needed.

Usage:
  # Start Ollama first:
  ollama serve
  ollama pull llama3.2

  python examples/06_ollama_local.py
"""

from __future__ import annotations

import asyncio

import hud
from hud.agents.ollama import OllamaAgent


async def main() -> None:
    # --- Agent: local Ollama model ---
    agent = OllamaAgent.create(model="llama3.2")

    # Check if Ollama is running
    if not await agent.check_health():
        print("Ollama server not reachable at http://localhost:11434")
        print("Start it with: ollama serve")
        return

    # Show available models
    models = await agent.list_models()
    print(f"Available models: {models}")

    # --- Environment: define tools the agent can use ---
    env = hud.Environment("demo")

    @env.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @env.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # --- Scenario: prompt + evaluation ---
    @env.scenario("math_challenge")
    async def math_challenge():
        yield "What is (7 + 3) * 5? Use the tools available to you. Give your final answer clearly."
        yield 1.0  # auto-pass for demo

    # --- Run ---
    task = env("math_challenge")

    print(f"Running agent with {agent.config.model}...")
    print("=" * 50)

    async with hud.eval(task, trace=False) as ctx:
        result = await agent.run(ctx, max_steps=10)

    print("=" * 50)
    print(f"Agent response: {result.content if result else 'No response'}")
    print(f"Reward: {ctx.reward}")


if __name__ == "__main__":
    asyncio.run(main())
