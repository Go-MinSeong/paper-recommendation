"""Auto-recommendation scheduler.

This module provides a scheduler that automatically sends paper recommendations
to users based on their configured intervals.
"""

import asyncio
from typing import Optional

from slack_sdk.web.async_client import AsyncWebClient

from config.logger import log
from config.settings import get_settings
from mcp_servers.auto_recommend.storage import AutoRecommendStorage
from mcp_servers.interest_manager.storage import InterestStorage
from src.recommender.engine import RecommendationEngine
from src.slack.formatters.blocks import (
    format_paper_thread_message,
    format_paper_summary_reply,
)


class AutoRecommendScheduler:
    """Scheduler for automatic paper recommendations.

    Checks for users due for recommendations and sends them automatically.

    Attributes:
        auto_recommend_storage: Storage for auto-recommend settings
        interest_storage: Storage for user interests
        recommendation_engine: Engine for generating recommendations
        slack_client: Slack Web API client
        check_interval: How often to check for due users (seconds)
    """

    def __init__(
        self,
        auto_recommend_storage: AutoRecommendStorage,
        interest_storage: InterestStorage,
        recommendation_engine: RecommendationEngine,
        slack_client: AsyncWebClient,
        check_interval: float = 60.0,  # Check every minute
    ) -> None:
        """Initialize auto-recommend scheduler.

        Args:
            auto_recommend_storage: Storage for auto-recommend settings
            interest_storage: Storage for user interests
            recommendation_engine: Engine for generating recommendations
            slack_client: Slack Web API client
            check_interval: How often to check for due users (seconds)
        """
        self.auto_recommend_storage = auto_recommend_storage
        self.interest_storage = interest_storage
        self.recommendation_engine = recommendation_engine
        self.slack_client = slack_client
        self.check_interval = check_interval
        self.settings = get_settings()

        self._task: Optional[asyncio.Task] = None
        self._running = False

        log.info(f"AutoRecommendScheduler initialized (check_interval={check_interval}s)")

    async def start(self) -> None:
        """Start the auto-recommend scheduler."""
        if self._running:
            log.warning("AutoRecommendScheduler is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        log.info("AutoRecommendScheduler started")

    async def stop(self) -> None:
        """Stop the auto-recommend scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        log.info("AutoRecommendScheduler stopped")

    async def _schedule_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                await self._check_and_send_recommendations()
            except Exception as e:
                log.error(f"Error in auto-recommend loop: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _check_and_send_recommendations(self) -> None:
        """Check for due users and send recommendations."""
        try:
            # Get users due for recommendations
            due_users = await self.auto_recommend_storage.get_due_users()

            if not due_users:
                log.debug("No users due for auto-recommendations")
                return

            log.info(f"Found {len(due_users)} users due for auto-recommendations")

            for user_settings in due_users:
                try:
                    await self._send_recommendations_to_user(user_settings.user_id, user_settings.paper_count)
                    await self.auto_recommend_storage.update_last_run(user_settings.user_id)
                except Exception as e:
                    log.error(
                        f"Failed to send auto-recommendations to user {user_settings.user_id}: {e}"
                    )

        except Exception as e:
            log.error(f"Error checking due users: {e}")

    async def _send_recommendations_to_user(self, user_id: str, paper_count: int) -> None:
        """Send recommendations to a specific user.

        Args:
            user_id: Slack user ID
            paper_count: Number of papers to recommend
        """
        # Get user's interest
        user_interest = await self.interest_storage.get(user_id)

        if not user_interest:
            log.warning(f"User {user_id} has auto-recommend enabled but no interest set")
            return

        log.info(f"Generating {paper_count} auto-recommendations for user {user_id}")

        # Temporarily override top_k for this recommendation
        original_top_k = self.recommendation_engine.top_k
        self.recommendation_engine.top_k = paper_count

        try:
            # Generate recommendations
            recommendations = await self.recommendation_engine.recommend(user_interest)

            if not recommendations:
                log.info(f"No new recommendations available for user {user_id}")
                return

            # Post each paper as a thread with summaries as reply
            for idx, rec in enumerate(recommendations, 1):
                # Format main thread message (title, date, citations, upvotes, link)
                thread_blocks = format_paper_thread_message(
                    rec=rec,
                    paper_index=idx,
                    total_papers=len(recommendations),
                )

                # Post main thread message
                thread_response = await self.slack_client.chat_postMessage(
                    channel=self.settings.slack_channel_id,
                    blocks=thread_blocks,
                    text=f"📄 [자동추천] {rec.title}",
                )

                thread_ts = thread_response.get("ts")

                # Format and post summaries as thread reply
                summary_blocks = format_paper_summary_reply(
                    rec=rec,
                    user_interest=user_interest,
                )

                await self.slack_client.chat_postMessage(
                    channel=self.settings.slack_channel_id,
                    thread_ts=thread_ts,
                    blocks=summary_blocks,
                    text=f"📝 {rec.title} - 요약",
                )

            log.info(
                f"Sent {len(recommendations)} auto-recommendations for user {user_id}"
            )

        finally:
            # Restore original top_k
            self.recommendation_engine.top_k = original_top_k

    async def run_now_for_user(self, user_id: str) -> int:
        """Manually trigger auto-recommend for a user.

        Args:
            user_id: Slack user ID

        Returns:
            int: Number of recommendations sent
        """
        settings = await self.auto_recommend_storage.get(user_id)

        if not settings:
            log.warning(f"No auto-recommend settings for user {user_id}")
            return 0

        await self._send_recommendations_to_user(user_id, settings.paper_count)
        await self.auto_recommend_storage.update_last_run(user_id)

        return settings.paper_count
