"""Data models for auto-recommendation settings.

This module defines Pydantic models for user auto-recommendation preferences.
"""

from datetime import datetime, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntervalUnit(str, Enum):
    """Interval unit for auto-recommendation scheduling."""

    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


class AutoRecommendSettings(BaseModel):
    """User-specific auto-recommendation settings.

    Attributes:
        user_id: Slack user ID
        enabled: Whether auto-recommend is enabled
        interval_value: Numeric value for interval (e.g., 1, 2, 7)
        interval_unit: Unit for interval (minutes, hours, days, weeks)
        paper_count: Number of papers to recommend each time
        preferred_time: Preferred time of day for recommendations (HH:MM)
        created_at: When settings were created
        updated_at: When settings were last updated
        last_run_at: When auto-recommend was last executed
    """

    user_id: str = Field(..., description="Slack user ID")
    enabled: bool = Field(default=True, description="Whether auto-recommend is enabled")
    interval_value: int = Field(default=1, ge=1, description="Interval numeric value")
    interval_unit: IntervalUnit = Field(
        default=IntervalUnit.DAYS,
        description="Interval unit",
    )
    paper_count: int = Field(default=3, ge=1, le=10, description="Papers per recommendation")
    preferred_time: Optional[str] = Field(
        default="09:00",
        description="Preferred time (HH:MM format)",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_run_at: Optional[datetime] = Field(default=None)

    def get_interval_seconds(self) -> int:
        """Calculate interval in seconds.

        Returns:
            int: Interval in seconds
        """
        multipliers = {
            IntervalUnit.MINUTES: 60,
            IntervalUnit.HOURS: 3600,
            IntervalUnit.DAYS: 86400,
            IntervalUnit.WEEKS: 604800,
        }
        return self.interval_value * multipliers[self.interval_unit]

    def get_interval_display(self) -> str:
        """Get human-readable interval string.

        Returns:
            str: Human-readable interval (e.g., "1일", "2시간", "1주일")
        """
        unit_names = {
            IntervalUnit.MINUTES: "분",
            IntervalUnit.HOURS: "시간",
            IntervalUnit.DAYS: "일",
            IntervalUnit.WEEKS: "주",
        }
        return f"{self.interval_value}{unit_names[self.interval_unit]}"

    def should_run_now(self) -> bool:
        """Check if auto-recommend should run now.

        Returns:
            bool: True if enough time has passed since last run
        """
        if not self.enabled:
            return False

        if self.last_run_at is None:
            return True

        elapsed = (datetime.now() - self.last_run_at).total_seconds()
        return elapsed >= self.get_interval_seconds()


class AllAutoRecommendSettings(BaseModel):
    """Container for all users' auto-recommend settings.

    Attributes:
        settings: Dictionary mapping user_id to their settings
        last_sync: Last synchronization timestamp
    """

    settings: dict[str, AutoRecommendSettings] = Field(default_factory=dict)
    last_sync: datetime = Field(default_factory=datetime.now)

    def get(self, user_id: str) -> Optional[AutoRecommendSettings]:
        """Get settings for a specific user.

        Args:
            user_id: Slack user ID

        Returns:
            AutoRecommendSettings or None if not found
        """
        return self.settings.get(user_id)

    def set(self, settings: AutoRecommendSettings) -> None:
        """Set or update settings for a user.

        Args:
            settings: User's auto-recommend settings
        """
        settings.updated_at = datetime.now()
        self.settings[settings.user_id] = settings
        self.last_sync = datetime.now()

    def remove(self, user_id: str) -> bool:
        """Remove settings for a user.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if settings were removed
        """
        if user_id in self.settings:
            del self.settings[user_id]
            self.last_sync = datetime.now()
            return True
        return False

    def get_due_users(self) -> list[AutoRecommendSettings]:
        """Get list of users whose auto-recommend is due.

        Returns:
            list[AutoRecommendSettings]: Users ready for auto-recommend
        """
        return [s for s in self.settings.values() if s.should_run_now()]

    def __len__(self) -> int:
        return len(self.settings)
