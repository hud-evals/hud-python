"""Task tracking for async telemetry operations.

This module provides infrastructure to track async tasks created during
telemetry operations (status updates, metric logging) to ensure they
complete before process shutdown, preventing telemetry loss.

The task tracker maintains strong references to tasks and explicitly cleans
them up when they complete via callbacks. This ensures tasks are not garbage
collected before they finish executing.

This is an internal module used by async context managers and cleanup
routines. Users typically don't interact with it directly.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module exports
__all__ = ["track_task", "wait_all_tasks", "TaskTracker"]

# Global singleton task tracker
_global_tracker: Optional["TaskTracker"] = None


class TaskTracker:
    """Tracks async tasks to ensure completion before shutdown.

    Uses a regular set with explicit cleanup to ensure tasks complete.
    Tasks are removed from the set when they finish via callback.
    """

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def track_task(self, coro, name: str = "task") -> asyncio.Task | None:
        """Create and track an async task.

        Args:
            coro: The coroutine to run
            name: Descriptive name for debugging and logging

        Returns:
            The created asyncio.Task, or None if no event loop is available
        """
        try:
            task = asyncio.create_task(coro, name=name)
            self._tasks.add(task)

            # Remove task from set when it completes and log any errors
            def cleanup_and_log(completed_task):
                self._tasks.discard(completed_task)
                if not completed_task.cancelled():
                    exc = completed_task.exception()
                    if exc:
                        logger.warning(f"Task '{name}' failed: {exc}")

            task.add_done_callback(cleanup_and_log)
            logger.debug(f"Tracking task '{name}' (total active: {len(self._tasks)})")
            return task

        except RuntimeError as e:
            # No event loop - fall back to fire_and_forget
            logger.warning(f"Cannot track task '{name}': {e}")
            from hud.utils.async_utils import fire_and_forget

            fire_and_forget(coro, name)
            return None

    async def wait_all(self, timeout: float = 30.0) -> int:
        """Wait for all tracked tasks to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            Number of tasks that were waited for
        """
        async with self._lock:
            # Get all pending tasks
            pending = [t for t in self._tasks if not t.done()]

            if not pending:
                logger.debug("No pending tasks to wait for")
                return 0

            logger.info(f"Waiting for {len(pending)} pending tasks...")

            try:
                # Wait with timeout
                done, still_pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=asyncio.ALL_COMPLETED
                )

                if still_pending:
                    logger.warning(f"{len(still_pending)} tasks still pending after {timeout}s")
                    # Cancel them
                    for task in still_pending:
                        task.cancel()

                logger.info(f"Completed {len(done)} tasks")
                return len(done)

            except Exception as e:
                logger.error(f"Error waiting for tasks: {e}")
                return 0

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return sum(1 for t in self._tasks if not t.done())


def get_global_tracker() -> TaskTracker:
    """Get or create the global task tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TaskTracker()
    return _global_tracker


def track_task(coro, name: str = "task") -> asyncio.Task | None:
    """Create and track an async task for telemetry operations.

    This is a convenience function that uses the global tracker to ensure
    the task completes before shutdown. Used internally by async context
    managers for status updates and metric logging.

    Args:
        coro: The coroutine to track
        name: Descriptive name for debugging

    Returns:
        The created task, or None if no event loop is available
    """
    tracker = get_global_tracker()
    return tracker.track_task(coro, name)


async def wait_all_tasks(timeout: float = 30.0) -> int:
    """Wait for all tracked telemetry tasks to complete.

    This ensures that all async telemetry operations (status updates, logs)
    complete before the calling function returns, preventing telemetry loss.

    Args:
        timeout: Maximum time to wait for tasks in seconds

    Returns:
        Number of tasks that completed
    """
    tracker = get_global_tracker()
    return await tracker.wait_all(timeout)
