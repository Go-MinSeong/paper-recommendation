"""Slack Block Kit message formatters.

This module provides functions to format messages using Slack's Block Kit
for rich, interactive message layouts.

Reference: https://api.slack.com/block-kit
"""

from datetime import datetime
from typing import Any

from mcp_servers.interest_manager.models import UserInterest
from src.recommender.engine import Recommendation


def format_recommendations_message(
    user_interest: UserInterest,
    recommendations: list[Recommendation],
) -> list[dict[str, Any]]:
    """Format personalized paper recommendations as Slack Block Kit message.

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
                    "text": f"추천 논문 {len(recommendations)}건 • <@{user_interest.user_id}>님을 위한 맞춤 추천",
                }
            ],
        }
    )

    blocks.append({"type": "divider"})

    # Each recommendation
    for idx, rec in enumerate(recommendations, 1):
        # Paper title and metadata
        similarity_pct = rec.similarity_score * 100
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{idx}. {rec.title}*\n\n"
                        f"_{rec.core_summary}_\n\n"
                        f"📊 유사도: {similarity_pct:.1f}% • 👍 {rec.upvotes} upvotes"
                    ),
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
