"""Remote HUD-hosted rollout placement."""

from __future__ import annotations

import warnings


class HostedRuntime:
    """HUD-hosted placement: the platform runs the whole rollout, agent included.

    Pass it as ``runtime=`` to ``Task.run`` or ``Taskset.run``. The agent travels as its
    registered type plus ``Agent.dump``, and ``poll_interval`` paces the wait for its
    trace. ``run_timeout`` is a deprecated alias for ``rollout_timeout``. A local cancel
    (Ctrl-C) or a deadline waits up to five seconds for the platform to accept the
    remote cancellation.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 5.0,
        run_timeout: float | None = None,
    ) -> None:
        if run_timeout is not None:
            warnings.warn(
                "HostedRuntime(run_timeout=...) is deprecated; pass "
                "rollout_timeout=... to Task.run or Taskset.run",
                DeprecationWarning,
                stacklevel=2,
            )
        self.poll_interval = poll_interval
        self.run_timeout = run_timeout
