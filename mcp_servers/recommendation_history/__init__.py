"""Recommendation history management module.

This module provides functionality for tracking and persisting
recommendation history to prevent duplicate recommendations.
"""

from mcp_servers.recommendation_history.models import RecommendationHistory, RecommendedPaper
from mcp_servers.recommendation_history.storage import (
    RecommendationHistoryStorage,
    RecommendationHistoryStorageError,
)

__all__ = [
    "RecommendationHistory",
    "RecommendedPaper",
    "RecommendationHistoryStorage",
    "RecommendationHistoryStorageError",
]
