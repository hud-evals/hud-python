"""Agent services for multi-turn conversations.

Chat is a thin client-side wrapper that accumulates message history and
runs a fresh agent.run() on each turn, passing the full conversation
as the scenario's input.
"""

from hud.services.chat import Chat

__all__ = ["Chat"]
