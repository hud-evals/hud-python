"""Task tracking for async telemetry operations.

This module provides infrastructure to track async tasks created during
telemetry operations (status updates, metric logging) to ensure they
complete before process shutdown, preventing telemetry loss.

The task tracker maintains strong references to tasks and explicitly cleans
them up when they complete via callbacks. This ensures tasks are not garbage
collected before they finish executing.

Thread Safety:
    Uses threading.Lock (not asyncio.Lock) because done callbacks run
    synchronously and need to modify the task set safely.

Race Condition Prevention:
    The wait_all() method uses a multi-pass approach to catch tasks that
    are created while waiting for existing tasks to complete.

This is an internal module used by async context managers and cleanup
routines. Users typically don't interact with it directly.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# Module exports
__all__ = ["TaskTracker", "track_task", "wait_all_tasks"]

# Global singleton task tracker
_global_tracker: "TaskTracker | None" = None


class TaskTracker:
    """Tracks async tasks to ensure completion before shutdown.

    Maintains a set of tasks with thread-safe access for both async code
    and synchronous callbacks. Tasks are automatically removed when they
    complete via done callbacks.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        # Use threading.Lock for synchronous access from done callbacks
        self._lock = threading.Lock()

    def track_task(self, coro: Coroutine[Any, Any, Any], name: str = "task") -> asyncio.Task | None:
        """Create and track an async task.

        Args:
            coro: The coroutine to run
            name: Descriptive name for debugging and logging

        Returns:
            The created asyncio.Task, or None if no event loop is available
        """
        try:
            task = asyncio.create_task(coro, name=name)

            # Add task to tracking set (thread-safe)
            with self._lock:
                self._tasks.add(task)
                task_count = len(self._tasks)

            # Setup cleanup callback
            def cleanup_callback(completed_task: asyncio.Task) -> None:
                """Remove completed task from tracking set and log failures."""
                with self._lock:
                    self._tasks.discard(completed_task)

                # Log exceptions outside lock to avoid blocking
                with contextlib.suppress(Exception):
                    if not completed_task.cancelled():
                        with contextlib.suppress(Exception):
                            exc = completed_task.exception()
                            if exc:
                                logger.warning("Task '%s' failed: %s", name, exc)

            task.add_done_callback(cleanup_callback)
            logger.debug("Tracking task '%s' (total active: %d)", name, task_count)
            return task

        except RuntimeError as e:
            # No event loop - fall back to fire_and_forget
            logger.warning("Cannot track task '%s': %s", name, e)
            from hud.utils.async_utils import fire_and_forget

            fire_and_forget(coro, name)
            return None

    async def wait_all(self, *, timeout_seconds: float = 30.0) -> int:
        """Wait for all tracked tasks to complete.

        Uses a multi-pass approach to handle race conditions where tasks are
        added while waiting for existing tasks to complete. This ensures that
        status updates created near the end of execution are still waited for.

        Args:
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            Number of tasks that completed
        """
        total_completed = 0
        time_remaining = timeout_seconds
        max_passes = 10  # Prevent infinite loops if tasks keep spawning

        for pass_num in range(max_passes):
            # Get snapshot of pending tasks (thread-safe)
            with self._lock:
                pending = [t for t in self._tasks if not t.done()]

            if not pending:
                if pass_num == 0:
                    logger.debug("No pending tasks to wait for")
                else:
                    logger.debug("All tasks completed after %d passes", pass_num)
                break

            # Log progress
            if pass_num == 0:
                logger.info("Waiting for %d pending tasks...", len(pending))
            else:
                logger.debug("Pass %d: Waiting for %d tasks", pass_num + 1, len(pending))

            # Wait for this batch (max 5s per pass to check for new tasks)
            batch_timeout = min(time_remaining, 5.0) if time_remaining > 0 else 5.0
            start_time = asyncio.get_event_loop().time()

            try:
                done, still_pending = await asyncio.wait(
                    pending, timeout=batch_timeout, return_when=asyncio.ALL_COMPLETED
                )
            except Exception as e:
                logger.error("Error waiting for tasks: %s", e)
                break

            # Update timing
            elapsed = asyncio.get_event_loop().time() - start_time
            time_remaining -= elapsed
            total_completed += len(done)

            # Handle timeout
            if still_pending:
                if time_remaining <= 0:
                    logger.warning(
                        "%d tasks still pending after %ss timeout - cancelling",
                        len(still_pending),
                        timeout_seconds,
                    )
                    for task in still_pending:
                        task.cancel()
                    break
                # Otherwise continue to next pass
            else:
                # All tasks from this batch completed, check for new ones
                with self._lock:
                    new_pending = [t for t in self._tasks if not t.done()]

                if not new_pending:
                    # No new tasks were added - we're done
                    break
                # Otherwise loop to wait for the new tasks

        if total_completed > 0:
            logger.info("Completed %d tasks", total_completed)

        return total_completed

    def get_pending_count(self) -> int:
        """Get number of pending tasks (thread-safe)."""
        with self._lock:
            return sum(1 for t in self._tasks if not t.done())


def get_global_tracker() -> TaskTracker:
    """Get or create the global task tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TaskTracker()
    return _global_tracker


def track_task(coro: Coroutine[Any, Any, Any], name: str = "task") -> asyncio.Task | None:
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

    Ensures that all async telemetry operations (status updates, logs)
    complete before the calling function returns, preventing telemetry loss.

    Uses a multi-pass approach to handle race conditions where status updates
    are created while waiting for other tasks to complete.

    Args:
        timeout_seconds: Maximum time to wait for tasks in seconds

    Returns:
        Number of tasks that completed
    """
    tracker = get_global_tracker()
    return await tracker.wait_all(timeout_seconds=timeout_seconds)
