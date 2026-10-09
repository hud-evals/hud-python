"""An actor that grades itself 0 and a verifier that reads its answer: 1 means the verifier won."""

from __future__ import annotations

from hud import Environment

actor = Environment("oracle-actor")
judge = Environment("oracle-judge")


@actor.template(id="relay")
async def relay(word: str):
    answer = yield f"Reply with this word and nothing else: {word}"
    yield {"score": 0.0, "answer": answer}


@judge.template(id="check-relay")
async def check_relay(expected: str):
    actor_result = yield ""
    yield 1.0 if (actor_result["answer"] or "").strip() == expected else 0.0


tasks = [relay(word="lighthouse")]
tasks[0].verifier = check_relay(expected="lighthouse")
