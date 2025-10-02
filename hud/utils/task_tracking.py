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
__all__ = ["TaskTracker", "track_task", "wait_all_tasks"]

# Global singleton task tracker
_global_tracker: Optional["TaskTracker"] = None


class TaskTracker:
    """Tracks async tasks to ensure completion before shutdown.

    Uses a regular set with explicit cleanup to ensure tasks complete.
    Tasks are removed from the set when they finish via callback.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def track_task(
        self, coro: asyncio.coroutines.Coroutine, name: str = "task"
    ) -> asyncio.Task | None:
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
            def cleanup_and_log(completed_task: asyncio.Task) -> None:
                self._tasks.discard(completed_task)
                if not completed_task.cancelled():
                    exc = completed_task.exception()
                    if exc:
                        logger.warning("Task '%s' failed: %s", name, exc)

            task.add_done_callback(cleanup_and_log)
            logger.debug("Tracking task '%s' (total active: %d)", name, len(self._tasks))
            return task

        except RuntimeError as e:
            # No event loop - fall back to fire_and_forget
            logger.warning("Cannot track task '%s': %s", name, e)
            from hud.utils.async_utils import fire_and_forget

            fire_and_forget(coro, name)
            return None

    async def wait_all(self, *, timeout_seconds: float = 30.0) -> int:
        """Wait for all tracked tasks to complete.

        Args:
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            Number of tasks that were waited for
        """
        async with self._lock:
            # Get all pending tasks
            pending = [t for t in self._tasks if not t.done()]

            if not pending:
                logger.debug("No pending tasks to wait for")
                return 0

            logger.info("Waiting for %d pending tasks...", len(pending))

            try:
                # Wait with timeout using asyncio.timeout context manager
                async with asyncio.timeout(timeout_seconds):
                    done, still_pending = await asyncio.wait(
                        pending, return_when=asyncio.ALL_COMPLETED
                    )

                    if still_pending:
                        logger.warning(
                            "%d tasks still pending after %ss", len(still_pending), timeout_seconds
                        )
                        # Cancel them
                        for task in still_pending:
                            task.cancel()

                    logger.info("Completed %d tasks", len(done))
                    return len(done)

            except TimeoutError:
                # Timeout occurred - cancel remaining tasks
                remaining = [t for t in pending if not t.done()]
                logger.warning("%d tasks timed out after %ss", len(remaining), timeout_seconds)
                for task in remaining:
                    task.cancel()
                # Return count of completed tasks
                completed = len([t for t in pending if t.done()])
                return completed
            except Exception as e:
                logger.error("Error waiting for tasks: %s", e)
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


def track_task(coro: asyncio.coroutines.Coroutine, name: str = "task") -> asyncio.Task | None:
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


async def wait_all_tasks(*, timeout_seconds: float = 30.0) -> int:
    """Wait for all tracked telemetry tasks to complete.

    This ensures that all async telemetry operations (status updates, logs)
    complete before the calling function returns, preventing telemetry loss.

    Args:
        timeout_seconds: Maximum time to wait for tasks in seconds

    Returns:
        Number of tasks that completed
    """
    tracker = get_global_tracker()
    return await tracker.wait_all(timeout_seconds=timeout_seconds)
