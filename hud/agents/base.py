"""Agent ABC: the rollout contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Self, cast

from typing_extensions import TypeVar

from hud.agents.types import AgentConfig

if TYPE_CHECKING:
    from hud.eval.run import Run
    from hud.utils.serialization import JsonObject

ConfigT_co = TypeVar("ConfigT_co", bound=AgentConfig, covariant=True, default=AgentConfig)


class Agent(ABC, Generic[ConfigT_co]):
    """Drives a live ``Run`` by recording its trajectory and final answer.

    Subclasses implement ``__call__(run)``; callers do ``await agent(run)``. Stateless
    per run — everything comes from ``run`` — so one instance drives many concurrent
    rollouts. The caller owns lifecycle status, cancellation, and grading.

    ``config`` is the agent's serializable identity: :meth:`dump` writes it and
    :meth:`load` rebuilds the agent from it. Subclasses parametrize ``Agent[XConfig]``,
    set ``config_cls = XConfig`` for construction, and, when they override
    ``__init__``, call ``super().__init__(config)``.
    """

    config_cls: ClassVar[type[AgentConfig]] = AgentConfig
    config: ConfigT_co

    def __init__(self, config: ConfigT_co | None = None) -> None:
        self.config = config or cast("ConfigT_co", self.config_cls())

    @abstractmethod
    async def __call__(self, run: Run) -> None:
        """Fill ``run.trace`` with the trajectory and final answer."""

    def dump(self) -> JsonObject:
        return self.config.model_dump(mode="json")

    @classmethod
    def load(cls, data: JsonObject) -> Self:
        return cls(cast("ConfigT_co", cls.config_cls.model_validate(data)))
