"""Error handling utilities for Slack handlers.

This module provides functions to convert various exceptions into
user-friendly Slack messages.
"""

from typing import Tuple

from config.logger import log


def format_error_message(error: Exception) -> Tuple[str, str]:
    """Convert exception to user-friendly message and error type.

    Args:
        error: The exception to convert

    Returns:
        Tuple of (error_type, user_message)
            - error_type: Classification of the error for logging
            - user_message: User-friendly error message

    Example:
        >>> error_type, msg = format_error_message(SummarizationError("Failed"))
        >>> print(msg)
        ❌ 요약 생성에 실패했습니다. 잠시 후 다시 시도해주세요.
    """
    error_class = error.__class__.__name__

    # Map exception types to user-friendly messages
    error_messages = {
        "SummarizationError": (
            "summarization_error",
            "❌ 요약 생성에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ),
        "InterestStorageError": (
            "storage_error",
            "❌ 관심사 저장에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ),
        "VectorStoreError": (
            "vector_store_error",
            "❌ 논문 검색에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ),
        "MilvusError": (
            "milvus_error",
            "❌ 데이터베이스 연결에 실패했습니다. 관리자에게 문의해주세요.",
        ),
        "EmbeddingError": (
            "embedding_error",
            "❌ 임베딩 생성에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ),
        "ValidationError": (
            "validation_error",
            "❌ 입력 값이 올바르지 않습니다. 다시 확인해주세요.",
        ),
        "FileNotFoundError": (
            "file_not_found",
            "❌ 필요한 파일을 찾을 수 없습니다. 관리자에게 문의해주세요.",
        ),
        "PermissionError": (
            "permission_error",
            "❌ 파일 접근 권한이 없습니다. 관리자에게 문의해주세요.",
        ),
        "ConnectionError": (
            "connection_error",
            "❌ 외부 서비스 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        ),
        "TimeoutError": (
            "timeout_error",
            "❌ 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
        ),
    }

    # Get specific error message or use generic one
    error_type, message = error_messages.get(
        error_class,
        (
            "unknown_error",
            "❌ 예기치 않은 오류가 발생했습니다. 관리자에게 문의해주세요.",
        ),
    )

    # Log the actual error for debugging
    log.error(
        f"Error in Slack handler: {error_type}",
        exc_info=True,
        extra={"error_class": error_class, "error_message": str(error)},
    )

    return error_type, message


def handle_command_error(error: Exception, command_name: str) -> str:
    """Handle errors from command execution and return user message.

    Args:
        error: The exception that occurred
        command_name: Name of the command that failed (e.g., "/insight")

    Returns:
        User-friendly error message string

    Example:
        >>> msg = handle_command_error(Exception("Failed"), "/insight")
        >>> print(msg)
        ❌ /insight 명령어 실행 중 오류가 발생했습니다.
    """
    error_type, base_message = format_error_message(error)

    log.error(
        f"Command {command_name} failed",
        extra={
            "command": command_name,
            "error_type": error_type,
            "error": str(error),
        },
    )

    return f"{base_message}\n\n_명령어: {command_name}_"


def format_validation_error(field_name: str, issue: str) -> str:
    """Format validation error message.

    Args:
        field_name: Name of the field that failed validation
        issue: Description of the validation issue

    Returns:
        Formatted error message

    Example:
        >>> msg = format_validation_error("관심사", "비어있을 수 없습니다")
        >>> print(msg)
        ❌ 입력 오류: 관심사 - 비어있을 수 없습니다
    """
    return f"❌ 입력 오류: {field_name} - {issue}"
