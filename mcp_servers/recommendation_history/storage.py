"""Storage manager for recommendation history.

This module provides persistent storage for recommendation history using JSON files.
"""

import json
from pathlib import Path

import aiofiles

from config.logger import log
from mcp_servers.recommendation_history.models import RecommendationHistory


class RecommendationHistoryStorageError(Exception):
    """Exception raised for recommendation history storage errors."""

    pass


class RecommendationHistoryStorage:
    """JSON-based storage for recommendation history.

    This class manages persistent storage of recommended papers
    using JSON files with async file operations.

    Attributes:
        storage_path: Path to the JSON storage file
    """

    def __init__(self, storage_path: str = "data/recommendation_history.json") -> None:
        """Initialize recommendation history storage.

        Args:
            storage_path: Path to storage file

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> await storage.add_recommendation("paper123", "Paper Title", "U123")
        """
        self.storage_path = Path(storage_path)
        self._ensure_directory()

        log.info(f"Recommendation history storage initialized: {self.storage_path}")

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, history: RecommendationHistory) -> None:
        """Save recommendation history to file.

        Args:
            history: RecommendationHistory object to save

        Raises:
            RecommendationHistoryStorageError: If save operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> history = RecommendationHistory()
            >>> history.add("paper123", "Title", "U123")
            >>> await storage.save(history)
        """
        try:
            log.debug(f"Saving {len(history)} recommendations to {self.storage_path}")

            # Convert set to list for JSON serialization
            data = history.model_dump(mode="json")

            async with aiofiles.open(self.storage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str))

            log.info(f"Successfully saved {len(history)} recommendations")

        except Exception as e:
            error_msg = f"Failed to save recommendation history: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

    async def load(self) -> RecommendationHistory:
        """Load recommendation history from file.

        Returns:
            RecommendationHistory: Loaded recommendation history

        Raises:
            RecommendationHistoryStorageError: If load operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> history = await storage.load()
            >>> print(len(history))
        """
        try:
            if not self.storage_path.exists():
                log.info("No existing recommendation history file, returning empty history")
                return RecommendationHistory()

            log.debug(f"Loading recommendation history from {self.storage_path}")

            async with aiofiles.open(self.storage_path, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            # Convert list back to set for recommended_paper_ids
            if "recommended_paper_ids" in data and isinstance(data["recommended_paper_ids"], list):
                data["recommended_paper_ids"] = set(data["recommended_paper_ids"])

            history = RecommendationHistory.model_validate(data)

            log.info(f"Successfully loaded {len(history)} recommendations")

            return history

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load recommendation history: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

    async def add_recommendation(self, paper_id: str, title: str, user_id: str) -> None:
        """Add a recommendation to history.

        Args:
            paper_id: Paper ID
            title: Paper title
            user_id: User ID who received the recommendation

        Raises:
            RecommendationHistoryStorageError: If operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> await storage.add_recommendation("paper123", "Paper Title", "U123")
        """
        try:
            history = await self.load()
            history.add(paper_id, title, user_id)
            await self.save(history)

            log.info(f"Added recommendation for paper {paper_id} to user {user_id}")

        except Exception as e:
            error_msg = f"Failed to add recommendation: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

    async def is_recommended(self, paper_id: str) -> bool:
        """Check if a paper has been recommended.

        Args:
            paper_id: Paper ID to check

        Returns:
            bool: True if the paper has been recommended

        Raises:
            RecommendationHistoryStorageError: If operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> is_rec = await storage.is_recommended("paper123")
        """
        try:
            history = await self.load()
            return history.is_recommended(paper_id)

        except Exception as e:
            error_msg = f"Failed to check recommendation: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

    async def get_recommended_ids(self) -> set[str]:
        """Get all recommended paper IDs.

        Returns:
            set[str]: Set of recommended paper IDs

        Raises:
            RecommendationHistoryStorageError: If operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> ids = await storage.get_recommended_ids()
        """
        try:
            history = await self.load()
            return history.recommended_paper_ids

        except Exception as e:
            error_msg = f"Failed to get recommended IDs: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e

    async def clear(self) -> None:
        """Clear all recommendation history.

        Raises:
            RecommendationHistoryStorageError: If operation fails

        Examples:
            >>> storage = RecommendationHistoryStorage()
            >>> await storage.clear()
        """
        try:
            history = RecommendationHistory()
            await self.save(history)
            log.info("Cleared all recommendation history")

        except Exception as e:
            error_msg = f"Failed to clear recommendation history: {str(e)}"
            log.error(error_msg)
            raise RecommendationHistoryStorageError(error_msg) from e
