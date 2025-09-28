"""Evaluation tools for browser environment."""

# Import modules to register @tool decorators
from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

from . import game_2048
from . import todo
from . import online_mind2web, webjudge_online_mind2web  # noqa: E402


__all__ = ["game_2048", "todo", "online_mind2web", "webjudge_online_mind2web"]
__all__.append("evaluate")