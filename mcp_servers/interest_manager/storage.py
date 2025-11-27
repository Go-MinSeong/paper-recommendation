"""Storage manager for user interests.

This module provides persistent storage for user interests using JSON files.
"""

import json
from pathlib import Path
from typing import Optional

import aiofiles

from config.logger import log
from mcp_servers.interest_manager.models import TeamInterests, UserInterest


class InterestStorageError(Exception):
    """Exception raised for interest storage errors."""

    pass


class InterestStorage:
    """JSON-based storage for user interests.

    This class manages persistent storage of team member interests
    using JSON files with async file operations.

    Attributes:
        storage_path: Path to the JSON storage file
    """

    def __init__(self, storage_path: str = "data/interests.json") -> None:
        """Initialize interest storage.

        Args:
            storage_path: Path to storage file

        Examples:
            >>> storage = InterestStorage()
            >>> await storage.save(team_interests)
        """
        self.storage_path = Path(storage_path)
        self._ensure_directory()

        log.info(f"Interest storage initialized: {self.storage_path}")

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, team_interests: TeamInterests) -> None:
        """Save team interests to file.

        Args:
            team_interests: TeamInterests object to save

        Raises:
            InterestStorageError: If save operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> team = TeamInterests()
            >>> team.add_interest("U123", "VLM research")
            >>> await storage.save(team)
        """
        try:
            log.debug(f"Saving {len(team_interests)} interests to {self.storage_path}")

            data = team_interests.model_dump(mode="json")

            async with aiofiles.open(self.storage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            log.info(f"Successfully saved {len(team_interests)} interests")

        except Exception as e:
            error_msg = f"Failed to save interests: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def load(self) -> TeamInterests:
        """Load team interests from file.

        Returns:
            TeamInterests: Loaded team interests

        Raises:
            InterestStorageError: If load operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> team = await storage.load()
            >>> print(len(team))
        """
        try:
            if not self.storage_path.exists():
                log.info("No existing interests file, returning empty TeamInterests")
                return TeamInterests()

            log.debug(f"Loading interests from {self.storage_path}")

            async with aiofiles.open(self.storage_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            team_interests = TeamInterests.model_validate(data)

            log.info(f"Successfully loaded {len(team_interests)} interests")

            return team_interests

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load interests: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def add_or_update(self, user_id: str, interest: str) -> UserInterest:
        """Add or update a user interest.

        Args:
            user_id: Slack user ID
            interest: Interest description

        Returns:
            UserInterest: Added or updated interest

        Raises:
            InterestStorageError: If operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> interest = await storage.add_or_update("U123", "VLM research")
        """
        try:
            team_interests = await self.load()
            user_interest = team_interests.add_interest(user_id, interest)
            await self.save(team_interests)

            log.info(f"Added/updated interest for user {user_id}")

            return user_interest

        except Exception as e:
            error_msg = f"Failed to add/update interest: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def get(self, user_id: str) -> Optional[UserInterest]:
        """Get user interest by user ID.

        Args:
            user_id: Slack user ID

        Returns:
            Optional[UserInterest]: User interest or None

        Raises:
            InterestStorageError: If operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> interest = await storage.get("U123")
        """
        try:
            team_interests = await self.load()
            return team_interests.get_interest(user_id)

        except Exception as e:
            error_msg = f"Failed to get interest: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def remove(self, user_id: str) -> bool:
        """Remove user interest.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if removed, False if not found

        Raises:
            InterestStorageError: If operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> removed = await storage.remove("U123")
        """
        try:
            team_interests = await self.load()
            removed = team_interests.remove_interest(user_id)

            if removed:
                await self.save(team_interests)
                log.info(f"Removed interest for user {user_id}")
            else:
                log.warning(f"Interest not found for user {user_id}")

            return removed

        except Exception as e:
            error_msg = f"Failed to remove interest: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def get_all(self) -> list[UserInterest]:
        """Get all user interests.

        Returns:
            list[UserInterest]: List of all interests

        Raises:
            InterestStorageError: If operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> interests = await storage.get_all()
        """
        try:
            team_interests = await self.load()
            return team_interests.get_all_interests()

        except Exception as e:
            error_msg = f"Failed to get all interests: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e

    async def clear(self) -> None:
        """Clear all interests.

        Raises:
            InterestStorageError: If operation fails

        Examples:
            >>> storage = InterestStorage()
            >>> await storage.clear()
        """
        try:
            team_interests = TeamInterests()
            await self.save(team_interests)
            log.info("Cleared all interests")

        except Exception as e:
            error_msg = f"Failed to clear interests: {str(e)}"
            log.error(error_msg)
            raise InterestStorageError(error_msg) from e
