"""Pytest configuration and fixtures.

This module provides shared fixtures for testing.
"""

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set test environment variables before importing settings
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["SLACK_BOT_TOKEN"] = "xoxb-test-token"
os.environ["SLACK_APP_TOKEN"] = "xapp-test-token"
os.environ["SLACK_CHANNEL_ID"] = "C12345678"
os.environ["ENVIRONMENT"] = "development"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI AsyncClient."""
    with patch("openai.AsyncOpenAI") as mock:
        client = AsyncMock()
        mock.return_value = client

        # Mock chat completions
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test summary content"
        client.chat.completions.create = AsyncMock(return_value=mock_completion)

        # Mock embeddings
        mock_embedding = MagicMock()
        mock_embedding.data = [MagicMock(embedding=[0.1] * 1536)]
        client.embeddings.create = AsyncMock(return_value=mock_embedding)

        yield client


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client."""
    with patch("pymilvus.connections") as mock_conn, \
         patch("pymilvus.utility") as mock_util, \
         patch("pymilvus.Collection") as mock_collection_class:

        # Mock connection
        mock_conn.connect = MagicMock()
        mock_conn.disconnect = MagicMock()

        # Mock utility
        mock_util.has_collection = MagicMock(return_value=True)

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.load = MagicMock()
        mock_collection.num_entities = 100
        mock_collection.insert = MagicMock(return_value=MagicMock(primary_keys=["id1", "id2"]))
        mock_collection.flush = MagicMock()
        mock_collection.query = MagicMock(return_value=[
            {"paper_id": "paper1"},
            {"paper_id": "paper2"},
        ])
        mock_collection.search = MagicMock(return_value=[[
            MagicMock(
                score=0.9,
                entity=MagicMock(
                    get=lambda key: {
                        "paper_id": "paper1",
                        "title": "Test Paper",
                        "abstract": "Test abstract",
                        "url": "https://example.com",
                        "upvotes": 10,
                    }.get(key)
                )
            )
        ]])

        mock_collection_class.return_value = mock_collection

        yield mock_collection


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing."""
    return {
        "paper_id": "test-paper-123",
        "title": "Test Paper: A Novel Approach to Testing",
        "abstract": "This paper presents a comprehensive approach to testing software systems.",
        "url": "https://arxiv.org/abs/2401.12345",
        "upvotes": 42,
        "score": 0.85,
    }


@pytest.fixture
def sample_user_interest():
    """Sample user interest for testing."""
    from mcp_servers.interest_manager.models import UserInterest
    return UserInterest(
        user_id="U12345678",
        interest="VLM을 이용한 객체 검출 및 CCTV 분석",
    )
