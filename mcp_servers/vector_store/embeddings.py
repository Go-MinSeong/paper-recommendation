"""OpenAI embeddings client for text vectorization.

This module provides interface to generate embeddings using OpenAI API.
"""

from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger import log
from config.settings import get_settings


class EmbeddingError(Exception):
    """Exception raised for embedding generation errors."""

    pass


class OpenAIEmbeddings:
    """OpenAI embeddings client.

    This client generates text embeddings using OpenAI's embedding models.

    Attributes:
        model: OpenAI embedding model name
        dimension: Embedding dimension size
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: Embedding model name (uses settings if not provided)
            dimension: Embedding dimension (uses settings if not provided)

        Examples:
            >>> embeddings = OpenAIEmbeddings()
            >>> vector = await embeddings.embed_text("Hello, world!")
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension

        self._client = AsyncOpenAI(api_key=self.api_key)

        log.info(f"Initialized OpenAI embeddings with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            list[float]: Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Examples:
            >>> embeddings = OpenAIEmbeddings()
            >>> vector = await embeddings.embed_text("Machine learning paper")
            >>> len(vector)
            1536
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            log.debug(f"Generating embedding for text (length: {len(text)})")

            response = await self._client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension,
            )

            embedding = response.data[0].embedding

            log.debug(f"Generated embedding with dimension: {len(embedding)}")

            return embedding

        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            log.error(error_msg)
            raise EmbeddingError(error_msg) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to embed

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts list is empty

        Examples:
            >>> embeddings = OpenAIEmbeddings()
            >>> texts = ["First paper", "Second paper"]
            >>> vectors = await embeddings.embed_texts(texts)
            >>> len(vectors)
            2
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        try:
            log.info(f"Generating embeddings for {len(valid_texts)} texts")

            response = await self._client.embeddings.create(
                model=self.model,
                input=valid_texts,
                dimensions=self.dimension,
            )

            embeddings = [item.embedding for item in response.data]

            log.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            log.error(error_msg)
            raise EmbeddingError(error_msg) from e

    async def close(self) -> None:
        """Close the OpenAI client.

        Examples:
            >>> embeddings = OpenAIEmbeddings()
            >>> await embeddings.close()
        """
        await self._client.close()
        log.debug("OpenAI embeddings client closed")

    async def __aenter__(self) -> "OpenAIEmbeddings":
        """Async context manager entry.

        Returns:
            OpenAIEmbeddings: Self instance
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
