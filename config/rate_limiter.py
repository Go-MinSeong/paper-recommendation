"""Rate limiter for API calls.

This module provides rate limiting functionality using asyncio Semaphore
to prevent exceeding API quotas.
"""

import asyncio
from functools import lru_cache

from config.logger import log
from config.settings import get_settings


class RateLimiter:
    """Rate limiter using asyncio Semaphore.

    This class provides a shared semaphore for limiting concurrent API calls.

    Attributes:
        semaphore: Asyncio Semaphore for rate limiting
        max_concurrent: Maximum concurrent requests allowed
    """

    def __init__(self, max_concurrent: int | None = None) -> None:
        """Initialize rate limiter.

        Args:
            max_concurrent: Maximum concurrent requests (uses settings if not provided)
        """
        settings = get_settings()
        self.max_concurrent = max_concurrent or settings.openai_max_concurrent_requests
        self._semaphore: asyncio.Semaphore | None = None

        log.info(f"RateLimiter initialized with max_concurrent={self.max_concurrent}")

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore.

        Returns:
            asyncio.Semaphore: The rate limiting semaphore

        Note:
            Semaphore is lazily created to ensure it's created in the correct event loop.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def acquire(self) -> None:
        """Acquire a slot from the semaphore."""
        await self.semaphore.acquire()
        log.debug(f"Rate limiter slot acquired (available: {self.available_slots})")

    def release(self) -> None:
        """Release a slot back to the semaphore."""
        self.semaphore.release()
        log.debug(f"Rate limiter slot released (available: {self.available_slots})")

    @property
    def available_slots(self) -> int:
        """Get the number of available slots.

        Returns:
            int: Number of available concurrent request slots
        """
        if self._semaphore is None:
            return self.max_concurrent
        return self._semaphore._value

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry - acquires semaphore."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - releases semaphore."""
        self.release()


# Global rate limiter instance for OpenAI API
_openai_rate_limiter: RateLimiter | None = None


def get_openai_rate_limiter() -> RateLimiter:
    """Get the global OpenAI rate limiter instance.

    Returns:
        RateLimiter: Global rate limiter for OpenAI API calls

    Examples:
        >>> limiter = get_openai_rate_limiter()
        >>> async with limiter:
        ...     await openai_api_call()
    """
    global _openai_rate_limiter
    if _openai_rate_limiter is None:
        _openai_rate_limiter = RateLimiter()
    return _openai_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _openai_rate_limiter
    _openai_rate_limiter = None
