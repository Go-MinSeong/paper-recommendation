"""Tests for rate limiter functionality.

Tests the RateLimiter class and OpenAI rate limiting.
"""

import asyncio
import time

import pytest

from config.rate_limiter import RateLimiter, get_openai_rate_limiter, reset_rate_limiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture(autouse=True)
    def reset_global_limiter(self):
        """Reset global rate limiter before each test."""
        reset_rate_limiter()
        yield
        reset_rate_limiter()

    def test_init_default(self):
        """Test default initialization."""
        limiter = RateLimiter(max_concurrent=5)
        assert limiter.max_concurrent == 5
        assert limiter.available_slots == 5

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test basic acquire and release."""
        limiter = RateLimiter(max_concurrent=2)

        await limiter.acquire()
        assert limiter.available_slots == 1

        await limiter.acquire()
        assert limiter.available_slots == 0

        limiter.release()
        assert limiter.available_slots == 1

        limiter.release()
        assert limiter.available_slots == 2

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        limiter = RateLimiter(max_concurrent=2)

        async with limiter:
            assert limiter.available_slots == 1

        assert limiter.available_slots == 2

    @pytest.mark.asyncio
    async def test_concurrent_limiting(self):
        """Test that concurrent requests are properly limited."""
        limiter = RateLimiter(max_concurrent=2)
        execution_order = []

        async def task(task_id: int):
            async with limiter:
                execution_order.append(f"start_{task_id}")
                await asyncio.sleep(0.1)
                execution_order.append(f"end_{task_id}")

        # Start 4 tasks with max 2 concurrent
        start_time = time.time()
        await asyncio.gather(
            task(1), task(2), task(3), task(4)
        )
        elapsed = time.time() - start_time

        # Should take at least 0.2 seconds (2 batches of 0.1s each)
        assert elapsed >= 0.2
        # All tasks should complete
        assert len(execution_order) == 8

    def test_get_openai_rate_limiter_singleton(self):
        """Test that get_openai_rate_limiter returns singleton."""
        limiter1 = get_openai_rate_limiter()
        limiter2 = get_openai_rate_limiter()

        assert limiter1 is limiter2

    def test_reset_rate_limiter(self):
        """Test resetting the global rate limiter."""
        limiter1 = get_openai_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_openai_rate_limiter()

        assert limiter1 is not limiter2
