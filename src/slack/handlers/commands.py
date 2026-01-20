"""Slack command handlers.

This module provides command handlers with dependency injection pattern.
Handlers acknowledge commands within 3 seconds and process them asynchronously.

Available commands:
- /set_interest: Register or update user interest
- /my_interest: View current interest
- /clear_interest: Remove interest
- /insight: Get personalized recommendations
- /history: View recommendation history
"""

import asyncio
from typing import Callable, Any

from slack_bolt.async_app import AsyncAck, AsyncRespond
from slack_sdk.web.async_client import AsyncWebClient

from config.logger import log
from config.settings import Settings
from src.recommender.engine import RecommendationEngine
from mcp_servers.interest_manager.storage import InterestStorage
from mcp_servers.recommendation_history.storage import RecommendationHistoryStorage
from src.slack.handlers.errors import handle_command_error, format_validation_error
from src.slack.formatters.blocks import (
    format_recommendations_message,
    format_interest_saved_message,
)


def create_set_interest_handler(
    interest_storage: InterestStorage,
) -> Callable:
    """Create /set_interest command handler with injected dependencies.

    Args:
        interest_storage: Storage for user interests

    Returns:
        Async command handler function

    Example:
        >>> storage = InterestStorage()
        >>> handler = create_set_interest_handler(storage)
        >>> app.command("/set_interest")(handler)
    """

    async def handle_set_interest(
        ack: AsyncAck,
        command: dict[str, Any],
        client: AsyncWebClient,
        respond: AsyncRespond,
    ) -> None:
        """Handle /set_interest command to save user interest.

        Args:
            ack: Acknowledge function (must be called within 3 seconds)
            command: Command payload from Slack
            client: Async Slack Web API client
            respond: Function to send response message
        """
        await ack()  # Acknowledge immediately

        user_id = command["user_id"]
        channel_id = command["channel_id"]
        interest_text = command.get("text", "").strip()

        log.info(
            f"Received /set_interest from user {user_id}",
            extra={"user_id": user_id, "interest": interest_text},
        )

        # Validate input
        if not interest_text:
            error_msg = format_validation_error(
                "관심사", "내용을 입력해주세요. 예: /set_interest VLM 연구"
            )
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=error_msg,
            )
            return

        # Process in background
        asyncio.create_task(
            _process_set_interest(
                user_id=user_id,
                channel_id=channel_id,
                interest_text=interest_text,
                client=client,
                storage=interest_storage,
            )
        )

    return handle_set_interest


async def _process_set_interest(
    user_id: str,
    channel_id: str,
    interest_text: str,
    client: AsyncWebClient,
    storage: InterestStorage,
) -> None:
    """Process set_interest request in background.

    Args:
        user_id: Slack user ID
        channel_id: Slack channel ID
        interest_text: User's interest description
        client: Async Slack Web API client
        storage: Interest storage instance
    """
    try:
        # Save interest
        user_interest = await storage.add_or_update(user_id, interest_text)

        log.info(
            f"Interest saved for user {user_id}",
            extra={
                "user_id": user_id,
                "interest": interest_text,
                "created_at": user_interest.created_at.isoformat(),
            },
        )

        # Send confirmation message
        message = format_interest_saved_message(
            interest=interest_text,
            created_at=user_interest.created_at,
        )

        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=message,
        )

    except Exception as e:
        error_msg = handle_command_error(e, "/set_interest")
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=error_msg,
        )


def create_insight_handler(
    recommendation_engine: RecommendationEngine,
    interest_storage: InterestStorage,
    settings: Settings,
) -> Callable:
    """Create /insight command handler with injected dependencies.

    Args:
        recommendation_engine: Engine for generating recommendations
        interest_storage: Storage for user interests
        settings: Application settings

    Returns:
        Async command handler function

    Example:
        >>> engine = RecommendationEngine(vector_store, summarizer)
        >>> storage = InterestStorage()
        >>> settings = get_settings()
        >>> handler = create_insight_handler(engine, storage, settings)
        >>> app.command("/insight")(handler)
    """

    async def handle_insight(
        ack: AsyncAck,
        command: dict[str, Any],
        client: AsyncWebClient,
        respond: AsyncRespond,
    ) -> None:
        """Handle /insight command to generate personalized recommendations.

        Args:
            ack: Acknowledge function (must be called within 3 seconds)
            command: Command payload from Slack
            client: Async Slack Web API client
            respond: Function to send response message
        """
        await ack()  # Acknowledge immediately

        user_id = command["user_id"]
        channel_id = command["channel_id"]

        log.info(
            f"Received /insight from user {user_id}",
            extra={"user_id": user_id},
        )

        # Process in background (may take several seconds)
        asyncio.create_task(
            _process_insight_request(
                user_id=user_id,
                channel_id=channel_id,
                client=client,
                engine=recommendation_engine,
                storage=interest_storage,
                settings=settings,
            )
        )

    return handle_insight


async def _process_insight_request(
    user_id: str,
    channel_id: str,
    client: AsyncWebClient,
    engine: RecommendationEngine,
    storage: InterestStorage,
    settings: Settings,
) -> None:
    """Process insight request in background with loading indicator.

    Args:
        user_id: Slack user ID
        channel_id: Slack channel ID
        client: Async Slack Web API client
        engine: Recommendation engine instance
        storage: Interest storage instance
        settings: Application settings
    """
    try:
        # Check if user has set interest
        user_interest = await storage.get(user_id)

        if not user_interest:
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=(
                    "⚠️ 먼저 관심사를 등록해주세요.\n\n"
                    "다음 명령어로 관심사를 등록할 수 있습니다:\n"
                    "`/set_interest <관심사 내용>`\n\n"
                    "예시: `/set_interest VLM을 이용한 CCTV 객체 검출`"
                ),
            )
            return

        # Show loading message (ephemeral messages cannot be deleted, so just inform user)
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="🔍 논문을 검색하고 요약하는 중입니다... 잠시만 기다려주세요.",
        )

        log.info(
            f"Generating recommendations for user {user_id}",
            extra={"user_id": user_id, "interest": user_interest.interest},
        )

        # Generate recommendations (may take 5-10 seconds)
        recommendations = await engine.recommend(user_interest)

        if not recommendations:
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=(
                    "⚠️ 추천할 논문을 찾지 못했습니다.\n\n"
                    "다음을 확인해주세요:\n"
                    "• 관심사가 너무 구체적이거나 좁지 않은지\n"
                    "• 최근 논문 데이터베이스가 업데이트되었는지\n\n"
                    "다른 관심사로 다시 시도하거나 관리자에게 문의해주세요."
                ),
            )
            return

        log.info(
            f"Generated {len(recommendations)} recommendations for user {user_id}",
            extra={
                "user_id": user_id,
                "recommendation_count": len(recommendations),
            },
        )

        # Format and post message to channel
        blocks = format_recommendations_message(
            user_interest=user_interest,
            recommendations=recommendations,
        )

        await client.chat_postMessage(
            channel=settings.slack_channel_id,  # Post to configured channel
            blocks=blocks,
            text=f"📚 {user_interest.interest}에 대한 추천 논문 {len(recommendations)}건",
        )

        # Send success message to user
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=f"✅ 추천 논문 {len(recommendations)}건을 <#{settings.slack_channel_id}> 채널에 게시했습니다!",
        )

    except Exception as e:
        log.exception(f"Error processing insight request for user {user_id}")
        error_msg = handle_command_error(e, "/insight")
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=error_msg,
        )


def create_my_interest_handler(
    interest_storage: InterestStorage,
) -> Callable:
    """Create /my_interest command handler.

    Args:
        interest_storage: Storage for user interests

    Returns:
        Async command handler function
    """

    async def handle_my_interest(
        ack: AsyncAck,
        command: dict[str, Any],
        client: AsyncWebClient,
        respond: AsyncRespond,
    ) -> None:
        """Handle /my_interest command to view current interest."""
        await ack()

        user_id = command["user_id"]
        channel_id = command["channel_id"]

        log.info(f"Received /my_interest from user {user_id}")

        try:
            user_interest = await interest_storage.get(user_id)

            if not user_interest:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=(
                        "📝 등록된 관심사가 없습니다.\n\n"
                        "`/set_interest <관심사>`로 등록해주세요.\n"
                        "예시: `/set_interest VLM을 이용한 객체 검출`"
                    ),
                )
                return

            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=(
                    f"📌 *등록된 관심사*\n\n"
                    f"> {user_interest.interest}\n\n"
                    f"등록일: {user_interest.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"수정일: {user_interest.updated_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"_관심사를 변경하려면 `/set_interest <새 관심사>`를 사용하세요._"
                ),
            )

        except Exception as e:
            error_msg = handle_command_error(e, "/my_interest")
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=error_msg,
            )

    return handle_my_interest


def create_clear_interest_handler(
    interest_storage: InterestStorage,
) -> Callable:
    """Create /clear_interest command handler.

    Args:
        interest_storage: Storage for user interests

    Returns:
        Async command handler function
    """

    async def handle_clear_interest(
        ack: AsyncAck,
        command: dict[str, Any],
        client: AsyncWebClient,
        respond: AsyncRespond,
    ) -> None:
        """Handle /clear_interest command to remove user interest."""
        await ack()

        user_id = command["user_id"]
        channel_id = command["channel_id"]

        log.info(f"Received /clear_interest from user {user_id}")

        try:
            removed = await interest_storage.remove(user_id)

            if removed:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="✅ 관심사가 삭제되었습니다.\n\n새로운 관심사를 등록하려면 `/set_interest <관심사>`를 사용하세요.",
                )
            else:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="⚠️ 등록된 관심사가 없습니다.",
                )

        except Exception as e:
            error_msg = handle_command_error(e, "/clear_interest")
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=error_msg,
            )

    return handle_clear_interest


def create_history_handler(
    recommendation_history: RecommendationHistoryStorage,
) -> Callable:
    """Create /history command handler.

    Args:
        recommendation_history: Storage for recommendation history

    Returns:
        Async command handler function
    """

    async def handle_history(
        ack: AsyncAck,
        command: dict[str, Any],
        client: AsyncWebClient,
        respond: AsyncRespond,
    ) -> None:
        """Handle /history command to view recommendation history."""
        await ack()

        user_id = command["user_id"]
        channel_id = command["channel_id"]

        log.info(f"Received /history from user {user_id}")

        try:
            history = await recommendation_history.load()

            # Filter history for this user
            user_recommendations = [
                detail for detail in history.details
                if detail.recommended_to_user_id == user_id
            ]

            if not user_recommendations:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=(
                        "📚 추천 이력이 없습니다.\n\n"
                        "`/insight` 명령어로 논문 추천을 받아보세요!"
                    ),
                )
                return

            # Format history (최근 10개만)
            recent = sorted(
                user_recommendations,
                key=lambda x: x.recommended_at,
                reverse=True
            )[:10]

            history_text = "📚 *최근 추천받은 논문*\n\n"
            for i, rec in enumerate(recent, 1):
                date_str = rec.recommended_at.strftime("%m/%d")
                history_text += f"{i}. [{date_str}] {rec.title[:60]}{'...' if len(rec.title) > 60 else ''}\n"

            history_text += f"\n_총 {len(user_recommendations)}건의 논문을 추천받았습니다._"

            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=history_text,
            )

        except Exception as e:
            error_msg = handle_command_error(e, "/history")
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=error_msg,
            )
