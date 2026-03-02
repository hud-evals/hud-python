"""External agent demo against HUD agent server.

Usage:
    # Terminal 1
    HUD_API_KEY=... HUD_ENV_NAME=... python examples/04_scenario_server.py

    # Terminal 2
    python examples/05_scenario_client.py
"""

from __future__ import annotations

import os

import httpx
from openai import OpenAI


def _stream_turn(client: OpenAI, *, model: str, session_id: str, prompt: str) -> None:
    print(f"Prompt: {prompt}")
    print("Answer: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        extra_headers={"X-HUD-Session-Id": session_id},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def main() -> None:
    base_url = os.getenv("HUD_AGENT_URL", "http://localhost:8321")
    model = os.getenv("HUD_MODEL", "gpt-4o")
    scenario_name = os.getenv("HUD_SCENARIO", "")

    print("Discovering scenarios...")
    response = httpx.get(f"{base_url}/scenarios", timeout=120.0)
    response.raise_for_status()
    scenarios = response.json()["scenarios"]
    if not scenarios:
        raise ValueError("No scenarios found on server")

    for scenario in scenarios:
        required = ", ".join(scenario["required_args"]) or "(none)"
        print(f"  {scenario['short_name']} - required: {required}")
    print()

    selected = next((s for s in scenarios if s["short_name"] == scenario_name), None)
    if selected is None:
        selected = scenarios[0]
        print(
            f"Scenario '{scenario_name}' not found; falling back to '{selected['short_name']}'"
        )

    default_args = {
        "id": "example-id",
        "task": "Investigate the issue and summarize findings.",
        "question": "What are the root causes?",
        "input_text": "Example input for scenario setup.",
    }
    scenario_args = {
        arg_name: default_args.get(arg_name, f"example-{arg_name}")
        for arg_name in selected.get("required_args", [])
    }
    print(f"Using scenario: {selected['short_name']}")

    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
        timeout=300.0,
    )

    first = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Begin and summarize the initial context."}],
        extra_body={
            "scenario": selected["short_name"],
            "scenario_args": scenario_args,
        },
    )
    first_answer = first.choices[0].message.content or ""
    hud_meta = getattr(first, "hud", None) or {}
    session_id = hud_meta.get("session_id", "") if isinstance(hud_meta, dict) else ""
    trace_url = hud_meta.get("trace_url", "") if isinstance(hud_meta, dict) else ""
    if not session_id:
        raise ValueError("No session_id returned from server")

    print(f"Session: {session_id}")
    print(f"Trace:   {trace_url}")
    print(f"Answer:  {first_answer[:200]}...\n")

    _stream_turn(client, model=model, session_id=session_id, prompt="What are the root causes?")
    _stream_turn(client, model=model, session_id=session_id, prompt="Give your top 3 recommendations.")

    finish = httpx.post(f"{base_url}/v1/sessions/{session_id}/finish", timeout=120.0)
    finish.raise_for_status()
    result = finish.json()
    print(f"Reward:    {result.get('reward')}")
    print(f"Trace URL: {result.get('trace_url')}")


if __name__ == "__main__":
    main()
