"""The featured Libbet visual-exploration task, bounded to 40 agent steps."""

from hud import Taskset

# `env` is re-exported so `hud eval tasks.py` can serve the Environment from here.
from env import env, play_game  # noqa: F401

explore = play_game(game="test", max_steps=40)
explore.slug = "libbet-explore-40"
explore.agent_config = {"max_steps": 40}

taskset = Taskset("libbet", [explore])
