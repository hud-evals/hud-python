"""Remote HUD-hosted rollout placement."""

from __future__ import annotations


class HostedRuntime:
    """HUD-hosted placement: the platform runs the whole rollout, agent included.

    Pass it as ``runtime=`` to ``Task.run`` or ``Taskset.run``. The agent travels as its
    registered type plus ``Agent.dump``, and ``poll_interval`` paces the wait for its
    trace. A local cancel (Ctrl-C) or a deadline waits up to five seconds for the
    platform to accept the remote cancellation.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 5.0,
    ) -> None:
        self.poll_interval = poll_interval
