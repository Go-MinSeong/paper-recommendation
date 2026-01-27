"""Slack Block Kit message formatters.

This module provides functions to format messages using Slack's Block Kit
for rich, interactive message layouts.

Reference: https://api.slack.com/block-kit
"""

from datetime import datetime
from typing import Any, Optional

from mcp_servers.interest_manager.models import UserInterest
from src.recommender.engine import Recommendation


def format_single_paper_message(
    rec: Recommendation,
    user_interest: UserInterest,
    paper_index: int,
    total_papers: int,
) -> list[dict[str, Any]]:
    """Format a single paper recommendation as Slack Block Kit message.

    Each paper is posted as an individual message to enable per-paper reactions.

    Args:
        rec: Single paper recommendation
        user_interest: User's interest with metadata
        paper_index: 1-based index of this paper
        total_papers: Total number of papers in this recommendation batch

    Returns:
        List of Block Kit blocks for a single paper

    Example:
        >>> rec = Recommendation(...)
        >>> interest = UserInterest(user_id="U123", interest="VLM 연구")
        >>> blocks = format_single_paper_message(rec, interest, 1, 3)
    """
    blocks: list[dict[str, Any]] = []

    # Build metadata line with publication date, citation count, and upvotes
    metadata_parts = []

    if rec.published_at:
        date_str = rec.published_at.strftime("%Y-%m-%d")
        metadata_parts.append(f"📅 {date_str}")

    if rec.citation_count is not None:
        metadata_parts.append(f"📖 인용 {rec.citation_count}회")

    if rec.upvotes is not None:
        metadata_parts.append(f"👍 {rec.upvotes}")

    metadata_line = " • ".join(metadata_parts) if metadata_parts else ""

    # Header with paper title
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📄 {rec.title[:145]}{'...' if len(rec.title) > 145 else ''}",
                "emoji": True,
            },
        }
    )

    # Context: metadata (date, citations)
    context_text = f"{paper_index}/{total_papers}"
    if metadata_line:
        context_text = f"{metadata_line} • {context_text}"

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": context_text,
                }
            ],
        }
    )

    blocks.append({"type": "divider"})

    # Core summary
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📝 핵심 요약*\n{rec.core_summary}",
            },
        }
    )

    # Contextualized summary
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*💡 맞춤 해석*\n{rec.contextualized_summary}",
            },
        }
    )

    # Action button
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📎 논문 읽기",
                        "emoji": True,
                    },
                    "url": rec.url,
                    "action_id": f"read_paper_{rec.paper_id}",
                    "style": "primary",
                }
            ],
        }
    )

    return blocks


def format_recommendation_header_message(
    user_interest: UserInterest,
    total_papers: int,
) -> list[dict[str, Any]]:
    """Format header message for a recommendation batch.

    This is posted as the first message, with individual papers as thread replies.

    Args:
        user_interest: User's interest with metadata
        total_papers: Total number of papers being recommended

    Returns:
        List of Block Kit blocks for the header message
    """
    blocks: list[dict[str, Any]] = []

    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📚 {user_interest.interest}",
                "emoji": True,
            },
        }
    )

    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"맞춤 논문 *{total_papers}*건을 추천합니다.\n"
                    f"각 논문에 반응을 남겨주시면 추천 품질 개선에 도움이 됩니다! 👍❤️🔥"
                ),
            },
        }
    )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                }
            ],
        }
    )

    return blocks


def format_recommendations_message(
    user_interest: UserInterest,
    recommendations: list[Recommendation],
) -> list[dict[str, Any]]:
    """Format personalized paper recommendations as Slack Block Kit message.

    DEPRECATED: Use format_single_paper_message for individual paper posts.

    Args:
        user_interest: User's interest with metadata
        recommendations: List of paper recommendations with summaries

    Returns:
        List of Block Kit blocks ready for Slack API

    Example:
        >>> interest = UserInterest(user_id="U123", interest="VLM 연구")
        >>> recs = [Recommendation(...), ...]
        >>> blocks = format_recommendations_message(interest, recs)
        >>> client.chat_postMessage(channel=channel_id, blocks=blocks)
    """
    blocks: list[dict[str, Any]] = []

    # Header
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📚 {user_interest.interest}",
                "emoji": True,
            },
        }
    )

    # Context: metadata
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"추천 논문 {len(recommendations)}건",
                }
            ],
        }
    )

    blocks.append({"type": "divider"})

    # Each recommendation
    for idx, rec in enumerate(recommendations, 1):
        # Build metadata line
        metadata_parts = []

        if rec.published_at:
            date_str = rec.published_at.strftime("%Y-%m-%d")
            metadata_parts.append(f"📅 {date_str}")

        if rec.citation_count is not None:
            metadata_parts.append(f"📖 인용 {rec.citation_count}회")

        if rec.upvotes is not None:
            metadata_parts.append(f"👍 {rec.upvotes}")

        metadata_line = " • ".join(metadata_parts) if metadata_parts else ""

        # Paper title and metadata
        text_content = f"*{idx}. {rec.title}*\n\n_{rec.core_summary}_"
        if metadata_line:
            text_content += f"\n\n{metadata_line}"

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text_content,
                },
            }
        )

        # Contextualized summary
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"💡 *맞춤 해석:*\n{rec.contextualized_summary}",
                },
            }
        )

        # Action button
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "논문 읽기",
                            "emoji": True,
                        },
                        "url": rec.url,
                        "action_id": f"read_paper_{rec.paper_id}",
                        "style": "primary",
                    }
                ],
            }
        )

        # Divider between papers
        if idx < len(recommendations):
            blocks.append({"type": "divider"})

    # Footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
                        "더 많은 추천을 받으려면 `/insight`를 사용하세요"
                    ),
                }
            ],
        }
    )

    return blocks


def format_interest_saved_message(
    interest: str,
    created_at: datetime,
) -> str:
    """Format interest saved confirmation message.

    Args:
        interest: User's interest description
        created_at: Timestamp when interest was saved

    Returns:
        Formatted message string

    Example:
        >>> msg = format_interest_saved_message("VLM 연구", datetime.now())
        >>> print(msg)
        ✅ 관심사가 저장되었습니다!
        ...
    """
    return (
        "✅ 관심사가 저장되었습니다!\n\n"
        f"*관심사:* `{interest}`\n"
        f"*등록 시간:* {created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "이제 `/insight` 명령어로 맞춤 논문 추천을 받아보실 수 있습니다."
    )


def format_error_message(error_type: str, details: str = "") -> str:
    """Format error message for users.

    Args:
        error_type: Type of error (e.g., "not_found", "api_error")
        details: Additional error details

    Returns:
        Formatted error message string

    Example:
        >>> msg = format_error_message("not_found", "논문 없음")
        >>> print(msg)
        ❌ 오류가 발생했습니다
        ...
    """
    base_messages = {
        "not_found": "요청한 리소스를 찾을 수 없습니다.",
        "api_error": "외부 API 호출에 실패했습니다.",
        "validation_error": "입력값이 올바르지 않습니다.",
        "storage_error": "데이터 저장에 실패했습니다.",
        "network_error": "네트워크 연결에 실패했습니다.",
    }

    base_msg = base_messages.get(error_type, "알 수 없는 오류가 발생했습니다.")

    message = f"❌ 오류가 발생했습니다\n\n*오류 유형:* {error_type}\n*설명:* {base_msg}"

    if details:
        message += f"\n*상세:* {details}"

    message += "\n\n잠시 후 다시 시도해주세요. 문제가 계속되면 관리자에게 문의해주세요."

    return message


def format_loading_message() -> str:
    """Format loading/processing message.

    Returns:
        Loading message string

    Example:
        >>> msg = format_loading_message()
        >>> print(msg)
        🔍 논문을 검색하고 요약하는 중입니다...
    """
    return "🔍 논문을 검색하고 요약하는 중입니다... 잠시만 기다려주세요."


def format_no_interest_message() -> str:
    """Format message when user has no interest set.

    Returns:
        Instruction message string

    Example:
        >>> msg = format_no_interest_message()
        >>> print(msg)
        ⚠️ 먼저 관심사를 등록해주세요.
        ...
    """
    return (
        "⚠️ 먼저 관심사를 등록해주세요.\n\n"
        "다음 명령어로 관심사를 등록할 수 있습니다:\n"
        "`/set_interest <관심사 내용>`\n\n"
        "예시: `/set_interest VLM을 이용한 CCTV 객체 검출`"
    )


def format_no_recommendations_message() -> str:
    """Format message when no recommendations are found.

    Returns:
        Information message string

    Example:
        >>> msg = format_no_recommendations_message()
        >>> print(msg)
        ⚠️ 추천할 논문을 찾지 못했습니다.
        ...
    """
    return (
        "⚠️ 추천할 논문을 찾지 못했습니다.\n\n"
        "다음을 확인해주세요:\n"
        "• 관심사가 너무 구체적이거나 좁지 않은지\n"
        "• 최근 논문 데이터베이스가 업데이트되었는지\n\n"
        "다른 관심사로 다시 시도하거나 관리자에게 문의해주세요."
    )
