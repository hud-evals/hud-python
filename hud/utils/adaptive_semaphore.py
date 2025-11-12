"""Adaptive semaphore for rate-limit-aware concurrency control."""

from __future__ import annotations

import asyncio
import logging
import math

logger = logging.getLogger(__name__)


class AdaptiveSemaphore:
    """Semaphore that can adapt its concurrency based on rate limit errors.
    
    This semaphore starts with an initial concurrency limit and automatically
    reduces it by 50% (ceiling) after detecting sustained rate limiting errors.
    
    Args:
        initial_value: Starting concurrency limit
        min_value: Minimum concurrency (won't reduce below this)
        rate_limit_threshold: Number of rate limit errors before reducing (default: 3)
    
    Example:
        >>> sem = AdaptiveSemaphore(initial_value=20, min_value=1)
        >>> async with sem:
        >>>     # Your concurrent work here
        >>>     # If rate limits occur, concurrency auto-reduces: 20 → 10 → 5 → 3 → 2 → 1
    """
    
    def __init__(
        self, 
        initial_value: int, 
        min_value: int = 1,
        rate_limit_threshold: int = 3
    ):
        self._initial_value = initial_value
        self._current_value = initial_value
        self._min_value = min_value
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._rate_limit_count = 0
        self._rate_limit_threshold = rate_limit_threshold
        
    async def __aenter__(self):
        """Acquire a slot, waiting if necessary."""
        # Wait until we can acquire
        while True:
            async with self._lock:
                if self._active_count < self._current_value:
                    self._active_count += 1
                    break
            await asyncio.sleep(0.1)  # Small delay before checking again
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release slot and handle rate limit errors."""
        # Release slot
        async with self._lock:
            self._active_count -= 1
        
        # Check if this was a rate limit error
        if exc_val and self._is_rate_limit_error(exc_val):
            await self._handle_rate_limit()
        
        return False
    
    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if exception is a rate limit error.
        
        Detects various rate limiting indicators:
        - HTTP 429 (Too Many Requests)
        - "rate limit" in error message
        - "quota" exceeded errors
        - "overloaded" API errors
        
        Args:
            exc: Exception to check
            
        Returns:
            True if this is a rate limit error
        """
        exc_str = str(exc).lower()
        return (
            "429" in exc_str or 
            "rate limit" in exc_str or 
            "quota" in exc_str or 
            "overloaded" in exc_str
        )
    
    async def _handle_rate_limit(self):
        """Handle rate limit by reducing concurrency by half (ceiling).
        
        After rate_limit_threshold errors (default 3), reduces concurrency
        by 50% using ceiling division. Won't reduce below min_value.
        """
        async with self._lock:
            self._rate_limit_count += 1
            
            if self._rate_limit_count >= self._rate_limit_threshold:
                if self._current_value > self._min_value:
                    old_value = self._current_value
                    # Reduce by ceiling of half
                    reduction = math.ceil(self._current_value / 2)
                    self._current_value = max(reduction, self._min_value)
                    
                    logger.warning(
                        f"🔽 Rate limit threshold reached ({self._rate_limit_count} errors). "
                        f"Reducing concurrency by ~50%: {old_value} → {self._current_value}"
                    )
                    # Reset counter
                    self._rate_limit_count = 0
                else:
                    logger.warning(
                        f"⚠️  At minimum concurrency ({self._min_value}), "
                        f"cannot reduce further despite rate limits"
                    )
    
    def get_current_value(self) -> int:
        """Get current concurrency limit.
        
        Returns:
            Current maximum number of concurrent operations
        """
        return self._current_value
    
    def get_initial_value(self) -> int:
        """Get initial concurrency limit.
        
        Returns:
            Starting maximum number of concurrent operations
        """
        return self._initial_value
    
    def get_reduction_count(self) -> int:
        """Get number of times concurrency was reduced.
        
        Returns:
            Number of reduction events
        """
        # Calculate based on how many halvings occurred
        if self._current_value == self._initial_value:
            return 0
        # Approximate number of reductions
        reductions = 0
        val = self._initial_value
        while val > self._current_value and val > self._min_value:
            val = max(math.ceil(val / 2), self._min_value)
            reductions += 1
        return reductions

