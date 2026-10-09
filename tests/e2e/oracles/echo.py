"""The agent repeats a word from the prompt, so a reward of 1 proves the prompt reached it."""

from __future__ import annotations

from hud import Environment

env = Environment("oracle-echo")


@env.template(id="echo")
async def echo(word: str):
    answer = yield f"Reply with this word and nothing else: {word}"
    yield 1.0 if (answer or "").strip() == word else 0.0


tasks = [echo(word="tangerine")]
