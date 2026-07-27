"""Tasks for blank — run with: hud eval tasks.py <agent>   (e.g. claude)."""

from env import count

# ``hud eval`` collects these Tasks — each is the ``count`` task bound to
# concrete args. Add your own, or build them in a loop.
tasks = [
    count(sentence="Strawberry world", letter="r"),
    count(sentence="banana", letter="a"),
]
