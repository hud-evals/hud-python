"""The integration contract: a foreign task format as a frontend to HUD.

An integration translates a foreign benchmark format into HUD's *what* —
:class:`~hud.eval.Taskset` rows and :class:`Environment` s — and never owns
execution: placement stays a format-agnostic execution-time concern
(:mod:`hud.eval.runtime`). No codegen roundtrip to run foreign tasks.

The contract is two verbs: load the format's data as rows, and construct
the live environment those rows join (by env name; each row's template id
dispatches within it). *Where* the constructor can execute
depends on the format: in-process formats (a dataset plus scoring code) run
it anywhere; formats whose environment is a container filesystem run it
inside an image built for that purpose — packaging the constructor into
such images is a format extra (e.g. ``harbor.adapt``), not part of the
contract.

Implementations live outside core — this repository's ``integrations/``, or
any installable package; core knows only this interface, and imports none of
them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from hud.eval import Taskset

    from .env import Environment


class Integration(ABC):
    """One foreign format's frontend. See module docstring for the contract."""

    #: The format's identifier, and the scheme its loaded tasksets are
    #: origin-stamped with (``<name>:<ref>``).
    name: ClassVar[str]

    @abstractmethod
    def load(self, ref: str | Path) -> Taskset:
        """Foreign data as rows, origin-stamped ``<name>:<ref>``."""

    @abstractmethod
    def environment(self, ref: str | Path, *, name: str | None = None) -> Environment:
        """The live environment serving *ref*'s tasks.

        *name* selects among several env groups when the ref has more than
        one. Rows join the env by name; each row's template id dispatches
        within it. Freshness is the placement's concern: providers call the
        constructor per acquisition.
        """
