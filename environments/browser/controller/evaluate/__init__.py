"""Evaluation tools for browser environment."""

# Import modules to register @tool decorators
from hud.tools.base import BaseHub

from . import game_2048
from . import todo
from . import autonomous_eval, webjudge_online_mind2web  # noqa: E402


__all__ = ["game_2048", "todo", "autonomous_eval", "webjudge_online_mind2web"]