"""The featured Libbet visual-exploration task, bounded to 40 agent steps."""

from hud import Taskset

from env import play_game

explore = play_game(game="test", max_steps=40)
explore.slug = "libbet-explore-40"
explore.agent_config = {"max_steps": 40}

taskset = Taskset("libbet", [explore])
