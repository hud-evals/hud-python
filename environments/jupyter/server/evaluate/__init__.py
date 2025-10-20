from __future__ import annotations
from hud.tools import BaseHub

evaluate = BaseHub("evaluate")

from . import dumb
from . import compare
from . import eval_single
from . import eval_all

__all__ = ["evaluate"]
