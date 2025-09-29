"""Global task tracking for async operations in HUD SDK.

This module provides a centralized way to track all async tasks created
during telemetry operations, ensuring they complete before process shutdown.
"""

import asyncio
import logging
from typing import Set, Optional
from weakref import WeakSet

logger = logging.getLogger(__name__)

# Global task tracker instance
_global_tracker: Optional['TaskTracker'] = None


class TaskTracker:
    """Tracks all async tasks to ensure completion before shutdown."""
    
    def __init__(self):
        # Use WeakSet to avoid keeping tasks alive
        self._tasks: WeakSet[asyncio.Task] = WeakSet()
        self._lock = asyncio.Lock()
    
    def track_task(self, coro, name: str = "task") -> asyncio.Task:
        """Create and track an async task.
        
        Args:
            coro: The coroutine to run
            name: Description of the task for debugging
            
        Returns:
            The created task
        """
        try:
            task = asyncio.create_task(coro, name=name)
            self._tasks.add(task)
            
            # Log errors from tasks
            def log_exception(task):
                if not task.cancelled():
                    exc = task.exception()
                    if exc:
                        logger.warning(f"Task '{name}' failed: {exc}")
            
            task.add_done_callback(log_exception)
            logger.debug(f"Tracking task '{name}', total active: {len(self._tasks)}")
            return task
            
        except RuntimeError as e:
            # No event loop
            logger.warning(f"Cannot track task '{name}': {e}")
            # Fall back to fire_and_forget
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
                    pending, 
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED
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


def track_task(coro, name: str = "task") -> asyncio.Task:
    """Track a task using the global tracker.
    
    This is a convenience function that uses the global tracker.
    """
    tracker = get_global_tracker()
    return tracker.track_task(coro, name)


async def wait_all_tasks(timeout: float = 30.0) -> int:
    """Wait for all globally tracked tasks.
    
    This is a convenience function that uses the global tracker.
    """
    tracker = get_global_tracker()
    return await tracker.wait_all(timeout)
