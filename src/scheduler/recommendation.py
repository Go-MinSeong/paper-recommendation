"""Scheduled recommendation service.

This module handles scheduled delivery of personalized paper recommendations
to team members based on their registered interests.
"""

import asyncio
from datetime import datetime

from slack_sdk.web.async_client import AsyncWebClient

from config.logger import log
from config.settings import get_settings
from mcp_servers.interest_manager.storage import InterestStorage
from src.recommender.engine import RecommendationEngine
from src.slack.formatters.blocks import format_recommendations_message


class RecommendationScheduler:
    """Scheduler for automated recommendation delivery.

    This scheduler periodically generates and sends personalized
    recommendations to all team members with registered interests.

    Attributes:
        engine: Recommendation engine instance
        interest_storage: Storage for user interests
        slack_client: Slack Web API client
        interval_hours: Interval between recommendation runs
        _running: Whether the scheduler is currently running
        _task: Background task reference
    """

    def __init__(
        self,
        engine: RecommendationEngine,
        interest_storage: InterestStorage,
        slack_client: AsyncWebClient,
        interval_hours: float = 24.0,
    ) -> None:
        """Initialize recommendation scheduler.

        Args:
            engine: Recommendation engine for generating recommendations
            interest_storage: Storage for user interests
            slack_client: Slack Web API client for posting messages
            interval_hours: Interval between runs (default: 24 hours)
        """
        self.engine = engine
        self.interest_storage = interest_storage
        self.slack_client = slack_client
        self.interval_hours = interval_hours
        self._running = False
        self._task: asyncio.Task | None = None

        log.info(
            f"Recommendation scheduler initialized: interval={interval_hours}h"
        )

    async def start(self, run_immediately: bool = False) -> None:
        """Start the scheduler.

        Args:
            run_immediately: Whether to run recommendations immediately on start
        """
        if self._running:
            log.warning("Recommendation scheduler is already running")
            return

        self._running = True
        log.info("Starting recommendation scheduler")

        if run_immediately:
            log.info("Running initial recommendation delivery...")
            await self.send_recommendations_to_all()

        # Start background task for periodic recommendations
        self._task = asyncio.create_task(self._schedule_loop())
        log.info("Recommendation scheduler background task started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            log.warning("Recommendation scheduler is not running")
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        log.info("Recommendation scheduler stopped")

    async def _schedule_loop(self) -> None:
        """Internal loop for scheduled recommendation delivery."""
        interval_seconds = self.interval_hours * 3600

        while self._running:
            try:
                log.debug(f"Next recommendation run in {self.interval_hours} hours")
                await asyncio.sleep(interval_seconds)

                if self._running:
                    await self.send_recommendations_to_all()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in recommendation scheduler loop: {e}")
                await asyncio.sleep(60)

    async def send_recommendations_to_all(self) -> int:
        """Send recommendations to all users with registered interests.

        Returns:
            int: Number of users who received recommendations
        """
        start_time = datetime.now()
        log.info("=" * 60)
        log.info("Starting scheduled recommendation delivery")
        log.info("=" * 60)

        settings = get_settings()
        successful_count = 0

        try:
            # Get all user interests and pick the most recently updated one
            all_interests = await self.interest_storage.get_all()

            if not all_interests:
                log.info("No users with registered interests found")
                return 0

            # Use only the most recently updated interest
            latest_interest = max(all_interests, key=lambda i: i.updated_at)
            log.info(
                f"Found {len(all_interests)} users with interests, "
                f"using latest from user {latest_interest.user_id}"
            )

            for interest in [latest_interest]:
                try:
                    log.info(f"Generating recommendations for user {interest.user_id}")

                    recommendations = await self.engine.recommend(interest)

                    if not recommendations:
                        log.info(f"No recommendations available for user {interest.user_id}")
                        continue

                    # Format and post message
                    blocks = format_recommendations_message(
                        user_interest=interest,
                        recommendations=recommendations,
                    )

                    await self.slack_client.chat_postMessage(
                        channel=settings.slack_channel_id,
                        blocks=blocks,
                        text=f"📚 <@{interest.user_id}>님을 위한 추천 논문 {len(recommendations)}건",
                    )

                    successful_count += 1
                    log.info(
                        f"Sent {len(recommendations)} recommendations to user {interest.user_id}"
                    )

                    # Small delay between users to avoid rate limiting
                    await asyncio.sleep(1)

                except Exception as e:
                    log.error(
                        f"Failed to send recommendations to user {interest.user_id}: {e}"
                    )
                    continue

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            log.info("=" * 60)
            log.info(f"Recommendation delivery completed in {elapsed:.1f}s")
            log.info(f"Users processed: {len(all_interests)}")
            log.info(f"Users with recommendations: {successful_count}")
            log.info("=" * 60)

            return successful_count

        except Exception as e:
            log.error(f"Scheduled recommendation delivery failed: {e}")
            return 0

    @property
    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return self._running
