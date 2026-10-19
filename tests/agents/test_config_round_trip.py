"""An agent's configuration survives the hosted submit: the platform rebuilds the same agent.

A hosted rollout sends the agent as its registered type and its dumped config;
the platform loads that type's agent from it. Each row submits an agent with
non-default settings and loads it back from what the platform received.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.eval import HostedRuntime, Task
from hud.types import AgentType
from tests.agents.families import FAMILIES, NotApplicable, over_families

if TYPE_CHECKING:
    from tests.support import Platform

CONFIGS: dict[str, dict[str, Any] | NotApplicable] = {
    "claude": {"model": "claude-sonnet-4-6", "max_steps": 5, "tool_timeout_seconds": 30},
    "openai": {"model": "gpt-5.6", "prompt_cache_key": "shared"},
    "gemini": {"model": "gemini-m", "max_output_tokens": 512},
    "openai_compatible": {"model": "m", "completion_kwargs": {"temperature": 0.2}},
    "claude_cli": {"reasoning_effort": "max", "max_steps": 3},
    "codex_cli": {"sandbox": "read-only", "reasoning_effort": "high"},
    "browser_use": NotApplicable("not a registered agent type; hosted execution refuses it"),
}


@pytest.mark.parametrize(("family", "config"), over_families(CONFIGS))
async def test_the_platform_rebuilds_the_submitted_agent_from_its_type_and_config(
    family: str, config: dict[str, Any], platform: Platform
) -> None:
    agent = FAMILIES[family].agent_cls(FAMILIES[family].config(gateway=True, **config))

    await HostedRuntime(poll_interval=0).run(
        Task(env="sums", id="add"), agent, job_id="2" * 32, trace_id="1" * 32
    )

    (trace,) = platform.traces.values()
    assert trace.submitted is not None
    spec = trace.submitted["agent"]
    rebuilt = AgentType(spec["type"]).cls.load(spec["config"])
    assert (type(rebuilt), rebuilt.dump()) == (type(agent), agent.dump())
