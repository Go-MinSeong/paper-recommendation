"""Vector store service integrating embeddings and Milvus.

This module provides high-level interface for storing and retrieving paper embeddings.
"""

from typing import Any

from config.logger import log
from mcp_servers.paper_collector.models import Paper, PaperCollection
from mcp_servers.vector_store.embeddings import OpenAIEmbeddings
from mcp_servers.vector_store.milvus_client import MilvusClient


class VectorStoreService:
    """Vector store service for paper embeddings.

    This service combines OpenAI embeddings and Milvus vector database
    to provide complete paper storage and retrieval functionality.

    Attributes:
        embeddings: OpenAI embeddings client
        milvus: Milvus database client
    """

    def __init__(self) -> None:
        """Initialize vector store service.

        Examples:
            >>> service = VectorStoreService()
            >>> await service.initialize()
        """
        self.embeddings = OpenAIEmbeddings()
        self.milvus = MilvusClient()

        log.info("Vector store service initialized")

    async def initialize(self) -> None:
        """Initialize connections to external services.

        Raises:
            Exception: If initialization fails

        Examples:
            >>> service = VectorStoreService()
            >>> await service.initialize()
        """
        try:
            log.info("Initializing vector store service")
            await self.milvus.connect()
            log.info("Vector store service ready")
        except Exception as e:
            log.error(f"Failed to initialize vector store service: {e}")
            raise

    async def close(self) -> None:
        """Close connections to external services.

        Examples:
            >>> service = VectorStoreService()
            >>> await service.initialize()
            >>> await service.close()
        """
        await self.embeddings.close()
        await self.milvus.disconnect()
        log.info("Vector store service closed")

    async def store_papers(self, papers: PaperCollection) -> list[str]:
        """Store papers with their embeddings in Milvus.

        Args:
            papers: Collection of papers to store

        Returns:
            list[str]: List of inserted IDs

        Raises:
            Exception: If storage fails

        Examples:
            >>> service = VectorStoreService()
            >>> await service.initialize()
            >>> papers = PaperCollection(papers=[...])
            >>> ids = await service.store_papers(papers)
        """
        if not papers or len(papers) == 0:
            log.warning("No papers to store")
            return []

        try:
            log.info(f"Storing {len(papers)} papers")

            # Generate embeddings for all papers
            texts = [paper.to_text() for paper in papers]
            embeddings = await self.embeddings.embed_texts(texts)

            # Prepare data for insertion
            paper_ids = [paper.id for paper in papers]
            titles = [paper.title for paper in papers]
            abstracts = [paper.abstract for paper in papers]
            urls = [str(paper.url) for paper in papers]
            upvotes = [paper.upvotes for paper in papers]

            # Insert into Milvus
            inserted_ids = await self.milvus.insert(
                paper_ids=paper_ids,
                titles=titles,
                abstracts=abstracts,
                urls=urls,
                embeddings=embeddings,
                upvotes=upvotes,
            )

            log.info(f"Successfully stored {len(inserted_ids)} papers")

            return inserted_ids

        except Exception as e:
            log.error(f"Failed to store papers: {e}")
            raise

    async def search_similar_papers(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search for papers similar to the query text.

        Args:
            query_text: Query text (e.g., user interest description)
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            list[dict[str, Any]]: List of similar papers with metadata

        Raises:
            Exception: If search fails

        Examples:
            >>> service = VectorStoreService()
            >>> await service.initialize()
            >>> results = await service.search_similar_papers(
            ...     query_text="VLM for object detection",
            ...     top_k=3
            ... )
        """
        try:
            log.info(f"Searching for papers similar to: '{query_text[:50]}...'")

            # Generate embedding for query
            query_embedding = await self.embeddings.embed_text(query_text)

            # Search in Milvus
            results = await self.milvus.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
            )

            log.info(f"Found {len(results)} similar papers")

            return results

        except Exception as e:
            log.error(f"Failed to search papers: {e}")
            raise

    async def __aenter__(self) -> "VectorStoreService":
        """Async context manager entry.

        Returns:
            VectorStoreService: Self instance
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
