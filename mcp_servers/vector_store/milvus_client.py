"""Milvus vector database client.

This module provides interface to interact with Milvus for vector storage and retrieval.
"""

from typing import Any, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from config.logger import log
from config.settings import get_settings


class MilvusError(Exception):
    """Exception raised for Milvus operation errors."""

    pass


class MilvusClient:
    """Milvus vector database client.

    This client manages connections to Milvus and provides methods for
    vector storage, search, and collection management.

    Attributes:
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to use
        dimension: Vector dimension size
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize Milvus client.

        Args:
            host: Milvus host (uses settings if not provided)
            port: Milvus port (uses settings if not provided)
            collection_name: Collection name (uses settings if not provided)
            dimension: Vector dimension (uses settings if not provided)

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
        """
        settings = get_settings()

        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.collection_name = collection_name or settings.milvus_collection_name
        self.dimension = dimension or settings.embedding_dimension

        self._collection: Optional[Collection] = None
        self._connected = False

        log.info(f"Initialized Milvus client: {self.host}:{self.port}")

    async def connect(self) -> None:
        """Connect to Milvus server.

        Raises:
            MilvusError: If connection fails

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
        """
        try:
            log.info(f"Connecting to Milvus at {self.host}:{self.port}")

            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
            )

            self._connected = True
            log.info("Successfully connected to Milvus")

            # Create or load collection
            await self._ensure_collection()

        except Exception as e:
            error_msg = f"Failed to connect to Milvus: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def disconnect(self) -> None:
        """Disconnect from Milvus server.

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
            >>> await client.disconnect()
        """
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            log.info("Disconnected from Milvus")

    async def _ensure_collection(self) -> None:
        """Ensure collection exists, create if not.

        Raises:
            MilvusError: If collection creation fails
        """
        try:
            if utility.has_collection(self.collection_name):
                log.info(f"Collection '{self.collection_name}' already exists")
                self._collection = Collection(name=self.collection_name)
            else:
                log.info(f"Creating collection '{self.collection_name}'")
                self._collection = await self._create_collection()

            # Load collection into memory
            self._collection.load()
            log.info(f"Collection '{self.collection_name}' loaded into memory")

            # Log collection statistics
            num_entities = self._collection.num_entities
            log.debug(f"[DB Stats] Collection '{self.collection_name}' contains {num_entities} papers")

        except Exception as e:
            error_msg = f"Failed to ensure collection: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def _create_collection(self) -> Collection:
        """Create new Milvus collection with schema.

        Returns:
            Collection: Created collection instance

        Raises:
            MilvusError: If collection creation fails
        """
        try:
            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True,
                    auto_id=False,
                ),
                FieldSchema(
                    name="paper_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                ),
                FieldSchema(
                    name="title",
                    dtype=DataType.VARCHAR,
                    max_length=1000,
                ),
                FieldSchema(
                    name="abstract",
                    dtype=DataType.VARCHAR,
                    max_length=5000,
                ),
                FieldSchema(
                    name="url",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                ),
                FieldSchema(
                    name="upvotes",
                    dtype=DataType.INT64,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Paper embeddings collection",
            )

            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default",
            )

            # Create IVF_FLAT index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }

            collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )

            log.info(f"Created collection '{self.collection_name}' with COSINE metric")

            return collection

        except Exception as e:
            error_msg = f"Failed to create collection: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def insert(
        self,
        paper_ids: list[str],
        titles: list[str],
        abstracts: list[str],
        urls: list[str],
        embeddings: list[list[float]],
        upvotes: list[int],
    ) -> list[str]:
        """Insert paper embeddings into Milvus.

        Args:
            paper_ids: List of paper IDs
            titles: List of paper titles
            abstracts: List of paper abstracts
            urls: List of paper URLs
            embeddings: List of embedding vectors
            upvotes: List of upvote counts

        Returns:
            list[str]: List of inserted IDs

        Raises:
            MilvusError: If insertion fails
            ValueError: If lists have different lengths

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
            >>> ids = await client.insert(
            ...     paper_ids=["p1"],
            ...     titles=["Title"],
            ...     abstracts=["Abstract"],
            ...     urls=["https://example.com"],
            ...     embeddings=[[0.1, 0.2, ...]],
            ...     upvotes=[10]
            ... )
        """
        if not self._collection:
            raise MilvusError("Collection not initialized. Call connect() first.")

        # Validate input lengths
        lengths = [len(paper_ids), len(titles), len(abstracts), len(urls), len(embeddings), len(upvotes)]
        if len(set(lengths)) != 1:
            raise ValueError(f"All input lists must have the same length, got {lengths}")

        try:
            # Generate unique IDs (paper_id + timestamp)
            import time
            timestamp = int(time.time() * 1000)
            ids = [f"{pid}_{timestamp}" for pid in paper_ids]

            data = [
                ids,
                paper_ids,
                titles,
                abstracts,
                urls,
                embeddings,
                upvotes,
            ]

            log.info(f"Inserting {len(paper_ids)} papers into Milvus")

            # Log papers being inserted
            for i, (pid, title) in enumerate(zip(paper_ids, titles)):
                log.debug(f"[Insert] Paper {i+1}: ID={pid}, Title='{title[:50]}...'")

            insert_result = self._collection.insert(data)

            # Flush to ensure data is persisted
            self._collection.flush()
            current_count = self._collection.num_entities
            log.debug(f"[Insert] Collection now contains {current_count} papers after insert")

            log.info(f"Successfully inserted {len(insert_result.primary_keys)} papers")

            return insert_result.primary_keys

        except Exception as e:
            error_msg = f"Failed to insert data: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search for similar papers using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold (0.0-1.0)

        Returns:
            list[dict[str, Any]]: List of search results with metadata

        Raises:
            MilvusError: If search fails

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
            >>> results = await client.search(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     top_k=3,
            ...     min_score=0.7
            ... )
        """
        if not self._collection:
            raise MilvusError("Collection not initialized. Call connect() first.")

        try:
            log.info(f"Searching for top {top_k} similar papers (min_score={min_score})")

            # Log query embedding info
            log.debug(f"[Search] Query embedding dimension: {len(query_embedding)}")
            log.debug(f"[Search] Query embedding sample (first 5): {query_embedding[:5]}")

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,  # Return exactly top_k after filtering
                output_fields=["paper_id", "title", "abstract", "url", "upvotes"],
            )

            # Log raw search results
            total_hits = sum(len(hits) for hits in results)
            log.debug(f"[Search] Raw results from Milvus: {total_hits} hits")

            # Filter and format results
            filtered_results = []
            filtered_out_count = 0
            for hits in results:
                for hit in hits:
                    log.debug(
                        f"[Search] Paper: '{hit.entity.get('title', 'N/A')[:50]}...' | "
                        f"Score: {hit.score:.4f} | "
                        f"Pass threshold: {hit.score >= min_score}"
                    )
                    if hit.score >= min_score:
                        filtered_results.append({
                            "paper_id": hit.entity.get("paper_id"),
                            "title": hit.entity.get("title"),
                            "abstract": hit.entity.get("abstract"),
                            "url": hit.entity.get("url"),
                            "upvotes": hit.entity.get("upvotes"),
                            "score": float(hit.score),
                        })
                    else:
                        filtered_out_count += 1

            # Sort by score and limit to top_k
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            filtered_results = filtered_results[:top_k]

            log.debug(f"[Search] Filtered out {filtered_out_count} papers below min_score={min_score}")
            log.info(f"Found {len(filtered_results)} papers matching criteria")

            return filtered_results

        except Exception as e:
            error_msg = f"Failed to search: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def get_existing_paper_ids(self) -> set[str]:
        """Get all existing paper IDs in the collection.

        Returns:
            set[str]: Set of paper IDs currently stored

        Raises:
            MilvusError: If query fails

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
            >>> existing_ids = await client.get_existing_paper_ids()
            >>> print(f"Found {len(existing_ids)} papers")
        """
        if not self._collection:
            raise MilvusError("Collection not initialized. Call connect() first.")

        try:
            log.debug("Querying existing paper IDs from Milvus")

            # Query all paper_ids from the collection
            results = self._collection.query(
                expr="",
                output_fields=["paper_id"],
                limit=16384,  # Milvus default max limit
            )

            paper_ids = {item["paper_id"] for item in results}

            log.info(f"Found {len(paper_ids)} existing papers in collection")

            return paper_ids

        except Exception as e:
            error_msg = f"Failed to get existing paper IDs: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def delete_collection(self) -> None:
        """Delete the collection.

        Raises:
            MilvusError: If deletion fails

        Examples:
            >>> client = MilvusClient()
            >>> await client.connect()
            >>> await client.delete_collection()
        """
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                log.info(f"Deleted collection '{self.collection_name}'")
            else:
                log.warning(f"Collection '{self.collection_name}' does not exist")

        except Exception as e:
            error_msg = f"Failed to delete collection: {str(e)}"
            log.error(error_msg)
            raise MilvusError(error_msg) from e

    async def __aenter__(self) -> "MilvusClient":
        """Async context manager entry.

        Returns:
            MilvusClient: Self instance
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
