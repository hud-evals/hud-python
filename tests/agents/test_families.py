"""Every agent in ``hud.agents`` is a family the agent contracts run over, or says why not."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import hud.agents
from hud.agents.base import Agent
from tests.agents.families import FAMILIES, NOT_FAMILIES


def test_every_agent_in_hud_agents_is_a_family_or_names_why_not() -> None:
    agents: set[str] = set()
    for info in pkgutil.walk_packages(hud.agents.__path__, "hud.agents."):
        if "tests" in info.name.split("."):
            continue
        module = importlib.import_module(info.name)
        agents |= {
            value.__name__
            for value in vars(module).values()
            if inspect.isclass(value)
            and issubclass(value, Agent)
            and value.__module__ == module.__name__
            and not inspect.isabstract(value)
        }

    families = {family.agent_cls.__name__ for family in FAMILIES.values()}
    assert agents == families | NOT_FAMILIES.keys()
