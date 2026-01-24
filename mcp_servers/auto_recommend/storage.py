"""Storage manager for auto-recommendation settings.

This module provides persistent storage for user auto-recommendation preferences.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles

from config.logger import log
from mcp_servers.auto_recommend.models import (
    AllAutoRecommendSettings,
    AutoRecommendSettings,
    IntervalUnit,
)


class AutoRecommendStorageError(Exception):
    """Exception raised for auto-recommend storage errors."""

    pass


class AutoRecommendStorage:
    """JSON-based storage for auto-recommendation settings.

    Attributes:
        storage_path: Path to the JSON storage file
    """

    def __init__(self, storage_path: str = "data/auto_recommend_settings.json") -> None:
        """Initialize auto-recommend storage.

        Args:
            storage_path: Path to storage file
        """
        self.storage_path = Path(storage_path)
        self._ensure_directory()
        log.info(f"Auto-recommend storage initialized: {self.storage_path}")

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, all_settings: AllAutoRecommendSettings) -> None:
        """Save all settings to file.

        Args:
            all_settings: All users' auto-recommend settings
        """
        try:
            data = all_settings.model_dump(mode="json")

            async with aiofiles.open(self.storage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str))

            log.debug(f"Saved {len(all_settings)} auto-recommend settings")

        except Exception as e:
            error_msg = f"Failed to save auto-recommend settings: {str(e)}"
            log.error(error_msg)
            raise AutoRecommendStorageError(error_msg) from e

    async def load(self) -> AllAutoRecommendSettings:
        """Load all settings from file.

        Returns:
            AllAutoRecommendSettings: All users' settings
        """
        try:
            if not self.storage_path.exists():
                log.info("No existing auto-recommend settings file")
                return AllAutoRecommendSettings()

            async with aiofiles.open(self.storage_path, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            all_settings = AllAutoRecommendSettings.model_validate(data)
            log.debug(f"Loaded {len(all_settings)} auto-recommend settings")

            return all_settings

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            log.error(error_msg)
            raise AutoRecommendStorageError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load auto-recommend settings: {str(e)}"
            log.error(error_msg)
            raise AutoRecommendStorageError(error_msg) from e

    async def get(self, user_id: str) -> Optional[AutoRecommendSettings]:
        """Get settings for a specific user.

        Args:
            user_id: Slack user ID

        Returns:
            AutoRecommendSettings or None
        """
        all_settings = await self.load()
        return all_settings.get(user_id)

    async def set(
        self,
        user_id: str,
        interval_value: int,
        interval_unit: IntervalUnit,
        paper_count: int = 3,
        preferred_time: Optional[str] = "09:00",
        enabled: bool = True,
    ) -> AutoRecommendSettings:
        """Set or update auto-recommend settings for a user.

        Args:
            user_id: Slack user ID
            interval_value: Interval numeric value
            interval_unit: Interval unit
            paper_count: Number of papers per recommendation
            preferred_time: Preferred time (HH:MM)
            enabled: Whether auto-recommend is enabled

        Returns:
            AutoRecommendSettings: Updated settings
        """
        all_settings = await self.load()

        # Check if user already has settings
        existing = all_settings.get(user_id)

        settings = AutoRecommendSettings(
            user_id=user_id,
            enabled=enabled,
            interval_value=interval_value,
            interval_unit=interval_unit,
            paper_count=paper_count,
            preferred_time=preferred_time,
            created_at=existing.created_at if existing else datetime.now(),
            last_run_at=existing.last_run_at if existing else None,
        )

        all_settings.set(settings)
        await self.save(all_settings)

        log.info(
            f"Updated auto-recommend settings for user {user_id}: "
            f"{settings.get_interval_display()}, {paper_count} papers"
        )

        return settings

    async def disable(self, user_id: str) -> bool:
        """Disable auto-recommend for a user.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if disabled successfully
        """
        all_settings = await self.load()
        settings = all_settings.get(user_id)

        if settings:
            settings.enabled = False
            settings.updated_at = datetime.now()
            all_settings.set(settings)
            await self.save(all_settings)
            log.info(f"Disabled auto-recommend for user {user_id}")
            return True

        return False

    async def enable(self, user_id: str) -> bool:
        """Enable auto-recommend for a user.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if enabled successfully
        """
        all_settings = await self.load()
        settings = all_settings.get(user_id)

        if settings:
            settings.enabled = True
            settings.updated_at = datetime.now()
            all_settings.set(settings)
            await self.save(all_settings)
            log.info(f"Enabled auto-recommend for user {user_id}")
            return True

        return False

    async def remove(self, user_id: str) -> bool:
        """Remove auto-recommend settings for a user.

        Args:
            user_id: Slack user ID

        Returns:
            bool: True if removed successfully
        """
        all_settings = await self.load()
        removed = all_settings.remove(user_id)

        if removed:
            await self.save(all_settings)
            log.info(f"Removed auto-recommend settings for user {user_id}")

        return removed

    async def update_last_run(self, user_id: str) -> None:
        """Update last run timestamp for a user.

        Args:
            user_id: Slack user ID
        """
        all_settings = await self.load()
        settings = all_settings.get(user_id)

        if settings:
            settings.last_run_at = datetime.now()
            all_settings.set(settings)
            await self.save(all_settings)
            log.debug(f"Updated last_run_at for user {user_id}")

    async def get_due_users(self) -> list[AutoRecommendSettings]:
        """Get list of users whose auto-recommend is due.

        Returns:
            list[AutoRecommendSettings]: Users ready for auto-recommend
        """
        all_settings = await self.load()
        return all_settings.get_due_users()
