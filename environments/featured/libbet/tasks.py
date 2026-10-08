"""The featured Libbet visual-exploration task, bounded to 40 agent steps."""

from hud import Taskset

# `env` is re-exported so `hud eval tasks.py` can serve the Environment from here.
from env import env, play_game  # noqa: F401

# Underscore-prefixed so the collector counts each task once (via the Taskset
# below), not also as a bare module global — that would double-count the slugs.
_test = play_game(game="test", max_steps=40)
_test.slug = "libbet-explore-40"
_test.agent_config = {"max_steps": 40}

taskset = Taskset("libbet", [_test])
