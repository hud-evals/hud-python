"""The agent repeats a word: a reward of 1 proves the prompt and the answer both arrived."""

from __future__ import annotations

from hud import Environment

env = Environment("echo")


@env.template()
async def echo(word: str):
    answer = yield f"Reply with: {word}"
    yield 1.0 if answer == word else 0.0


@env.template(returns=int)
async def count(word: str):
    answer = yield f"How many letters are in {word}?"
    yield 1.0 if answer.content == len(word) else 0.0
