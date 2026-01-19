"""Application configuration settings.

This module provides centralized configuration management using pydantic-settings.
All configuration values are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and type safety.

    All settings are loaded from environment variables or .env file.
    Required settings will raise an error if not provided.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Slack Configuration
    slack_bot_token: str = Field(..., description="Slack bot token (xoxb-...)")
    slack_app_token: str = Field(..., description="Slack app token (xapp-...)")
    slack_channel_id: str = Field(..., description="Slack channel ID for posting")

    # Milvus Configuration
    milvus_host: str = Field(default="localhost", description="Milvus host address")
    milvus_port: int = Field(default=19530, description="Milvus port number")
    milvus_collection_name: str = Field(
        default="paper_embeddings",
        description="Milvus collection name for paper embeddings",
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding dimension size",
    )

    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model for summarization",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation",
    )
    llm_max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens for LLM generation",
    )

    # Paper Collection Configuration
    paper_collection_limit: int = Field(
        default=30,
        gt=0,
        le=100,
        description="Number of papers to fetch from API per collection",
    )
    collection_interval_hours: float = Field(
        default=24.0,
        gt=0,
        description="Interval between paper collections in hours",
    )
    papers_fetch_schedule: str = Field(
        default="0 9 * * 1",
        description="Cron schedule for paper fetching (default: Monday 9AM)",
    )

    # Recommendation Configuration
    top_k_recommendations: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Number of top papers to recommend",
    )
    min_similarity_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for recommendations",
    )

    # Application Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )

    @field_validator("slack_bot_token")
    @classmethod
    def validate_slack_bot_token(cls, v: str) -> str:
        """Validate Slack bot token format."""
        if not v.startswith("xoxb-"):
            raise ValueError("Slack bot token must start with 'xoxb-'")
        return v

    @field_validator("slack_app_token")
    @classmethod
    def validate_slack_app_token(cls, v: str) -> str:
        """Validate Slack app token format."""
        if not v.startswith("xapp-"):
            raise ValueError("Slack app token must start with 'xapp-'")
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        """Validate embedding dimension is positive."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    @property
    def milvus_uri(self) -> str:
        """Get Milvus connection URI.

        Returns:
            str: Milvus connection URI in format 'host:port'

        Examples:
            >>> settings = Settings()
            >>> settings.milvus_uri
            'localhost:19530'
        """
        return f"{self.milvus_host}:{self.milvus_port}"

    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            bool: True if environment is production, False otherwise

        Examples:
            >>> settings = Settings(environment="production")
            >>> settings.is_production()
            True
        """
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            bool: True if environment is development, False otherwise

        Examples:
            >>> settings = Settings(environment="development")
            >>> settings.is_development()
            True
        """
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.

    Returns:
        Settings: Application settings instance

    Examples:
        >>> settings = get_settings()
        >>> print(settings.milvus_host)
        'localhost'
    """
    return Settings()
