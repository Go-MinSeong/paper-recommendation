"""Tests for recommendation history storage.

Tests the RecommendationHistoryStorage class functionality.
"""

import json
from pathlib import Path

import pytest

from mcp_servers.recommendation_history.models import RecommendationHistory
from mcp_servers.recommendation_history.storage import RecommendationHistoryStorage


class TestRecommendationHistoryModels:
    """Tests for recommendation history models."""

    def test_recommendation_history_add(self):
        """Test adding recommendations to history."""
        history = RecommendationHistory()

        history.add("paper1", "Test Paper 1", "U123")
        history.add("paper2", "Test Paper 2", "U456")

        assert len(history) == 2
        assert history.is_recommended("paper1")
        assert history.is_recommended("paper2")
        assert not history.is_recommended("paper3")

    def test_recommendation_history_duplicate(self):
        """Test that duplicate paper IDs are handled correctly."""
        history = RecommendationHistory()

        history.add("paper1", "Test Paper 1", "U123")
        history.add("paper1", "Test Paper 1 Again", "U456")

        # Set should only have one entry
        assert len(history.recommended_paper_ids) == 1
        # Details should have two entries
        assert len(history.details) == 2


class TestRecommendationHistoryStorage:
    """Tests for recommendation history storage."""

    @pytest.fixture
    def storage(self, temp_data_dir):
        """Create a temporary storage instance."""
        storage_path = temp_data_dir / "recommendation_history.json"
        return RecommendationHistoryStorage(storage_path=str(storage_path))

    @pytest.mark.asyncio
    async def test_load_empty(self, storage):
        """Test loading when no file exists."""
        history = await storage.load()

        assert len(history) == 0
        assert len(history.recommended_paper_ids) == 0

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage):
        """Test saving and loading history."""
        history = RecommendationHistory()
        history.add("paper1", "Test Paper 1", "U123")
        history.add("paper2", "Test Paper 2", "U456")

        await storage.save(history)
        loaded = await storage.load()

        assert len(loaded) == 2
        assert loaded.is_recommended("paper1")
        assert loaded.is_recommended("paper2")

    @pytest.mark.asyncio
    async def test_add_recommendation(self, storage):
        """Test adding a single recommendation."""
        await storage.add_recommendation("paper1", "Test Paper", "U123")

        history = await storage.load()
        assert history.is_recommended("paper1")

    @pytest.mark.asyncio
    async def test_get_recommended_ids(self, storage):
        """Test getting all recommended IDs."""
        await storage.add_recommendation("paper1", "Test 1", "U123")
        await storage.add_recommendation("paper2", "Test 2", "U456")

        ids = await storage.get_recommended_ids()

        assert "paper1" in ids
        assert "paper2" in ids
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_is_recommended(self, storage):
        """Test checking if paper is recommended."""
        await storage.add_recommendation("paper1", "Test 1", "U123")

        assert await storage.is_recommended("paper1")
        assert not await storage.is_recommended("paper2")

    @pytest.mark.asyncio
    async def test_clear(self, storage):
        """Test clearing all history."""
        await storage.add_recommendation("paper1", "Test 1", "U123")
        await storage.add_recommendation("paper2", "Test 2", "U456")

        await storage.clear()
        history = await storage.load()

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_json_serialization(self, storage, temp_data_dir):
        """Test that JSON file is properly formatted."""
        await storage.add_recommendation("paper1", "Test Paper", "U123")

        storage_path = temp_data_dir / "recommendation_history.json"
        with open(storage_path) as f:
            data = json.load(f)

        assert "recommended_paper_ids" in data
        assert "details" in data
        assert "paper1" in data["recommended_paper_ids"]
