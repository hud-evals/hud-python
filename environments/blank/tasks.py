"""Example tasks for the blank environment."""

from env import count, env

tasks = [
    count(sentence="Strawberry world", letter="r"),
    count(sentence="banana", letter="a"),
]

__all__ = ["env", "tasks"]
