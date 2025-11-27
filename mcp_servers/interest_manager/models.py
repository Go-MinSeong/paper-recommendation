"""Data models for user interests.

This module defines Pydantic models for managing user interests.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserInterest(BaseModel):
    """User interest data model.

    Attributes:
        user_id: Slack user ID
        interest: Interest description text
        embedding: Optional pre-computed embedding vector
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    user_id: str = Field(..., min_length=1, description="Slack user ID")
    interest: str = Field(..., min_length=1, description="Interest description")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Pre-computed embedding vector",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class TeamInterests(BaseModel):
    """Collection of team member interests.

    Attributes:
        interests: Dictionary mapping user_id to UserInterest
        last_sync: Last synchronization timestamp
    """

    interests: dict[str, UserInterest] = Field(
        default_factory=dict,
        description="Dictionary of user interests",
    )
    last_sync: datetime = Field(
        default_factory=datetime.now,
        description="Last sync timestamp",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def add_interest(self, user_id: str, interest: str) -> UserInterest:
        """Add or update user interest.

        Args:
            user_id: Slack user ID
            interest: Interest description

        Returns:
            UserInterest: Created or updated interest

        Examples:
            >>> team = TeamInterests()
            >>> interest = team.add_interest("U123", "VLM research")
            >>> print(interest.user_id)
            'U123'
        """
        if user_id in self.interests:
            # Update existing interest
            self.interests[user_id].interest = interest
            self.interests[user_id].updated_at = datetime.now()
            self.interests[user_id].embedding = None  # Reset embedding
        else:
            # Create new interest
            self.interests[user_id] = UserInterest(
                user_id=user_id,
                interest=interest,
            )

        self.last_sync = datetime.now()
        return self.interests[user_id]

    def get_interest(self, user_id: str) -> Optional[UserInterest]:
        """Get user interest by user ID.

        Args:
            user_id: Slack user ID

        Returns:
            Optional[UserInterest]: User interest or None if not found

        Examples:
            >>> team = TeamInterests()
            >>> team.add_interest("U123", "VLM research")
            >>> interest = team.get_interest("U123")
            >>> print(interest.interest)
            'VLM research'
        """
        return self.interests.get(user_id)

    def remove_interest(self, user_id: str) -> bool:
        """Remove user interest.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if removed, False if not found

        Examples:
            >>> team = TeamInterests()
            >>> team.add_interest("U123", "VLM research")
            >>> team.remove_interest("U123")
            True
        """
        if user_id in self.interests:
            del self.interests[user_id]
            self.last_sync = datetime.now()
            return True
        return False

    def get_all_interests(self) -> list[UserInterest]:
        """Get all user interests.

        Returns:
            list[UserInterest]: List of all interests

        Examples:
            >>> team = TeamInterests()
            >>> team.add_interest("U123", "VLM research")
            >>> team.add_interest("U456", "Object detection")
            >>> len(team.get_all_interests())
            2
        """
        return list(self.interests.values())

    def __len__(self) -> int:
        """Get number of interests.

        Returns:
            int: Number of interests
        """
        return len(self.interests)
