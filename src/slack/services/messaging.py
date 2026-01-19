"""Slack messaging service for encapsulating message sending logic.

This module provides high-level functions for common messaging patterns,
abstracting away the details of the Slack Web API.
"""

from typing import Any, Optional

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from config.logger import log
from mcp_servers.interest_manager.models import UserInterest
from src.recommender.engine import Recommendation
from src.slack.formatters.blocks import format_recommendations_message


async def post_recommendations(
    client: AsyncWebClient,
    channel_id: str,
    user_interest: UserInterest,
    recommendations: list[Recommendation],
) -> dict[str, Any]:
    """Post personalized paper recommendations to a Slack channel.

    Args:
        client: Async Slack Web API client
        channel_id: Target channel ID
        user_interest: User's interest with metadata
        recommendations: List of paper recommendations

    Returns:
        Slack API response dictionary

    Raises:
        SlackApiError: If message posting fails

    Example:
        >>> await post_recommendations(
        ...     client=client,
        ...     channel_id="C123456",
        ...     user_interest=interest,
        ...     recommendations=recs
        ... )
    """
    try:
        blocks = format_recommendations_message(user_interest, recommendations)

        response = await client.chat_postMessage(
            channel=channel_id,
            blocks=blocks,
            text=f"📚 {user_interest.interest}에 대한 추천 논문 {len(recommendations)}건",
        )

        log.info(
            "Posted recommendations to channel",
            extra={
                "channel_id": channel_id,
                "user_id": user_interest.user_id,
                "recommendation_count": len(recommendations),
                "message_ts": response.get("ts"),
            },
        )

        return response

    except SlackApiError as e:
        log.error(
            f"Failed to post recommendations: {e.response['error']}",
            exc_info=True,
            extra={
                "channel_id": channel_id,
                "error_code": e.response.get("error"),
            },
        )
        raise


async def post_ephemeral_message(
    client: AsyncWebClient,
    channel_id: str,
    user_id: str,
    text: str,
    blocks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Post an ephemeral message visible only to a specific user.

    Args:
        client: Async Slack Web API client
        channel_id: Channel where message appears
        user_id: User who can see the message
        text: Message text
        blocks: Optional Block Kit blocks

    Returns:
        Slack API response dictionary

    Raises:
        SlackApiError: If message posting fails

    Example:
        >>> await post_ephemeral_message(
        ...     client=client,
        ...     channel_id="C123456",
        ...     user_id="U123456",
        ...     text="✅ Success!"
        ... )
    """
    try:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "user": user_id,
            "text": text,
        }

        if blocks:
            kwargs["blocks"] = blocks

        response = await client.chat_postEphemeral(**kwargs)

        log.debug(
            "Posted ephemeral message",
            extra={
                "channel_id": channel_id,
                "user_id": user_id,
                "message_ts": response.get("message_ts"),
            },
        )

        return response

    except SlackApiError as e:
        log.error(
            f"Failed to post ephemeral message: {e.response['error']}",
            exc_info=True,
            extra={
                "channel_id": channel_id,
                "user_id": user_id,
                "error_code": e.response.get("error"),
            },
        )
        raise


async def delete_message(
    client: AsyncWebClient,
    channel_id: str,
    message_ts: str,
) -> dict[str, Any]:
    """Delete a message from a Slack channel.

    Args:
        client: Async Slack Web API client
        channel_id: Channel containing the message
        message_ts: Timestamp of the message to delete

    Returns:
        Slack API response dictionary

    Raises:
        SlackApiError: If message deletion fails

    Example:
        >>> response = await client.chat_postMessage(...)
        >>> await delete_message(
        ...     client=client,
        ...     channel_id="C123456",
        ...     message_ts=response["ts"]
        ... )
    """
    try:
        response = await client.chat_delete(
            channel=channel_id,
            ts=message_ts,
        )

        log.debug(
            "Deleted message",
            extra={
                "channel_id": channel_id,
                "message_ts": message_ts,
            },
        )

        return response

    except SlackApiError as e:
        # Don't raise for message_not_found errors (already deleted)
        if e.response.get("error") == "message_not_found":
            log.debug(
                "Message already deleted",
                extra={"channel_id": channel_id, "message_ts": message_ts},
            )
            return {"ok": True}

        log.error(
            f"Failed to delete message: {e.response['error']}",
            exc_info=True,
            extra={
                "channel_id": channel_id,
                "message_ts": message_ts,
                "error_code": e.response.get("error"),
            },
        )
        raise


async def update_message(
    client: AsyncWebClient,
    channel_id: str,
    message_ts: str,
    text: str,
    blocks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Update an existing message in a Slack channel.

    Args:
        client: Async Slack Web API client
        channel_id: Channel containing the message
        message_ts: Timestamp of the message to update
        text: New message text
        blocks: Optional new Block Kit blocks

    Returns:
        Slack API response dictionary

    Raises:
        SlackApiError: If message update fails

    Example:
        >>> response = await client.chat_postMessage(...)
        >>> await update_message(
        ...     client=client,
        ...     channel_id="C123456",
        ...     message_ts=response["ts"],
        ...     text="Updated text"
        ... )
    """
    try:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "ts": message_ts,
            "text": text,
        }

        if blocks:
            kwargs["blocks"] = blocks

        response = await client.chat_update(**kwargs)

        log.debug(
            "Updated message",
            extra={
                "channel_id": channel_id,
                "message_ts": message_ts,
            },
        )

        return response

    except SlackApiError as e:
        log.error(
            f"Failed to update message: {e.response['error']}",
            exc_info=True,
            extra={
                "channel_id": channel_id,
                "message_ts": message_ts,
                "error_code": e.response.get("error"),
            },
        )
        raise
