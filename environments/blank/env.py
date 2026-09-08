"""Minimal exact-match environment."""

from collections.abc import AsyncGenerator

from hud import Environment

env = Environment(name="blank")


@env.template(id="count")
async def count(sentence: str, letter: str) -> AsyncGenerator[str | float, str | None]:
    """Ask the agent to count a letter and grade the answer."""
    answer = yield f"How many times does '{letter}' appear in: '{sentence}'?"
    correct = str(sentence.lower().count(letter.lower()))
    yield float((answer or "").strip() == correct)
