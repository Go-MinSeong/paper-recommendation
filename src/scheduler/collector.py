"""Paper collection scheduler.

This module handles scheduled collection of papers from Hugging Face
and Semantic Scholar, storing them in Milvus vector database.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional

from config.logger import log
from mcp_servers.paper_collector.huggingface_api import (
    HuggingFacePapersClient,
    HuggingFacePapersAPIError,
)
from mcp_servers.paper_collector.semantic_scholar_api import (
    SemanticScholarClient,
    SemanticScholarAPIError,
    DEFAULT_TRENDING_QUERIES,
)
from mcp_servers.vector_store.service import VectorStoreService


class PaperSource(str, Enum):
    """Available paper sources."""

    HUGGINGFACE = "huggingface"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    BOTH = "both"


class PaperCollectionScheduler:
    """Scheduler for paper collection tasks.

    This scheduler periodically fetches papers from Hugging Face
    and/or Semantic Scholar and stores them in the vector database.

    Quality filtering:
        - min_upvotes: Minimum upvotes required for HuggingFace (default: 10)
        - min_citations: Minimum citation count for Semantic Scholar (default: 10)
        - max_age_days: Maximum paper age in days for Semantic Scholar (default: 365)

    Note:
        HuggingFace uses weekly trending papers automatically (current week).
        Semantic Scholar uses max_age_days for filtering.

    Attributes:
        vector_store: Vector store service instance
        interval_hours: Collection interval in hours
        paper_limit: Maximum number of papers to fetch per collection
        min_upvotes: Minimum upvotes filter (HuggingFace)
        min_citations: Minimum citation count (Semantic Scholar)
        max_age_days: Maximum paper age filter (Semantic Scholar only)
        source: Paper source (huggingface, semantic_scholar, or both)
        _running: Whether the scheduler is currently running
        _task: Background task reference
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        interval_hours: float = 24.0,
        paper_limit: int = 100,
        min_upvotes: int = 10,
        min_citations: int = 10,
        max_age_days: int = 365,
        source: PaperSource = PaperSource.BOTH,
        trending_queries: Optional[list[str]] = None,
    ) -> None:
        """Initialize paper collection scheduler.

        Args:
            vector_store: Vector store service for storing papers
            interval_hours: Interval between collections (default: 24 hours)
            paper_limit: Maximum papers to fetch per collection (default: 100)
            min_upvotes: Minimum upvotes required for HuggingFace (default: 10)
            min_citations: Minimum citation count for Semantic Scholar (default: 10)
            max_age_days: Maximum paper age in days for Semantic Scholar (default: 365)
            source: Paper source (default: both)
            trending_queries: Custom queries for Semantic Scholar (default: AI/ML queries)

        Note:
            HuggingFace automatically fetches weekly trending papers (current week).
            max_age_days only applies to Semantic Scholar.

        Examples:
            >>> scheduler = PaperCollectionScheduler(
            ...     vector_store=vector_store,
            ...     interval_hours=24.0,
            ...     paper_limit=100,
            ...     min_upvotes=10,
            ...     min_citations=10,
            ...     source=PaperSource.BOTH,
            ... )
        """
        self.vector_store = vector_store
        self.interval_hours = interval_hours
        self.paper_limit = paper_limit
        self.min_upvotes = min_upvotes
        self.min_citations = min_citations
        self.max_age_days = max_age_days
        self.source = source
        self.trending_queries = trending_queries or DEFAULT_TRENDING_QUERIES
        self._running = False
        self._task: Optional[asyncio.Task] = None

        log.info(
            f"Paper collection scheduler initialized: "
            f"interval={interval_hours}h, limit={paper_limit}, "
            f"min_upvotes={min_upvotes}, min_citations={min_citations}, "
            f"max_age_days={max_age_days}, source={source.value}"
        )

    async def start(self, run_immediately: bool = True) -> None:
        """Start the scheduler.

        Args:
            run_immediately: Whether to collect papers immediately on start

        Examples:
            >>> await scheduler.start(run_immediately=True)
        """
        if self._running:
            log.warning("Scheduler is already running")
            return

        self._running = True
        log.info("Starting paper collection scheduler")

        if run_immediately:
            log.info("Running initial paper collection...")
            await self.collect_papers()

        # Start background task for periodic collection
        self._task = asyncio.create_task(self._schedule_loop())
        log.info("Scheduler background task started")

    async def stop(self) -> None:
        """Stop the scheduler.

        Examples:
            >>> await scheduler.stop()
        """
        if not self._running:
            log.warning("Scheduler is not running")
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        log.info("Paper collection scheduler stopped")

    async def _schedule_loop(self) -> None:
        """Internal loop for scheduled paper collection."""
        interval_seconds = self.interval_hours * 3600

        while self._running:
            try:
                # Wait for next collection interval
                log.debug(f"Next collection in {self.interval_hours} hours")
                await asyncio.sleep(interval_seconds)

                if self._running:
                    await self.collect_papers()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in scheduler loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Brief pause before retry

    async def collect_papers(self) -> int:
        """Collect papers from configured sources and store in vector database.

        Returns:
            int: Number of papers successfully stored

        Examples:
            >>> count = await scheduler.collect_papers()
            >>> print(f"Collected {count} papers")
        """
        start_time = datetime.now()
        log.info("=" * 60)
        log.info(f"Starting paper collection task (source={self.source.value})")
        log.info("=" * 60)

        all_papers = []
        total_stored = 0

        try:
            # Step 1: Collect from HuggingFace if enabled
            if self.source in (PaperSource.HUGGINGFACE, PaperSource.BOTH):
                hf_papers = await self._collect_from_huggingface()
                all_papers.extend(hf_papers)

            # Step 2: Collect from Semantic Scholar if enabled
            if self.source in (PaperSource.SEMANTIC_SCHOLAR, PaperSource.BOTH):
                ss_papers = await self._collect_from_semantic_scholar()
                all_papers.extend(ss_papers)

            if not all_papers:
                log.warning("No papers fetched from any source")
                return 0

            # Deduplicate by paper ID
            seen_ids = set()
            unique_papers = []
            for paper in all_papers:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    unique_papers.append(paper)

            log.info(f"Total unique papers after deduplication: {len(unique_papers)}")

            # Limit to paper_limit
            unique_papers = unique_papers[: self.paper_limit]

            # Step 3: Store papers in vector database
            log.info("Storing papers in vector database with embeddings")
            inserted_ids = await self.vector_store.store_papers(unique_papers)
            total_stored = len(inserted_ids)

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            log.info("=" * 60)
            log.info(f"Paper collection completed in {elapsed:.1f}s")
            log.info(f"Papers fetched: {len(all_papers)}")
            log.info(f"Papers unique: {len(unique_papers)}")
            log.info(f"Papers stored: {total_stored}")
            log.info("=" * 60)

            return total_stored

        except Exception as e:
            log.error(f"Paper collection task failed: {e}")
            return total_stored

    async def _collect_from_huggingface(self) -> list:
        """Collect weekly trending papers from HuggingFace.

        Uses huggingface_hub library to fetch weekly trending papers.
        Papers are filtered by min_upvotes and sorted by trending score.

        Returns:
            List of Paper objects from HuggingFace
        """
        try:
            log.info(
                f"Fetching weekly trending papers from HuggingFace "
                f"(limit={self.paper_limit}, min_upvotes={self.min_upvotes})"
            )
            async with HuggingFacePapersClient() as client:
                papers = await client.fetch_papers(
                    limit=self.paper_limit,
                    min_upvotes=self.min_upvotes,
                )

            if papers and len(papers) > 0:
                log.info(f"Fetched {len(papers)} weekly trending papers from HuggingFace")
                for i, paper in enumerate(papers.papers[:5], 1):
                    log.debug(
                        f"  [HF-{i}] {paper.title[:60]}... "
                        f"(id={paper.id}, upvotes={paper.upvotes})"
                    )
                return list(papers.papers)
            else:
                log.warning("No papers fetched from HuggingFace")
                return []

        except HuggingFacePapersAPIError as e:
            log.error(f"Failed to fetch papers from HuggingFace: {e}")
            return []

    async def _collect_from_semantic_scholar(self) -> list:
        """Collect trending papers from Semantic Scholar.

        Returns:
            List of Paper objects from Semantic Scholar
        """
        try:
            log.info(
                f"Fetching trending papers from Semantic Scholar "
                f"(queries={len(self.trending_queries)}, min_citations={self.min_citations}, "
                f"max_age_days={self.max_age_days})"
            )
            async with SemanticScholarClient(timeout=30.0) as client:
                papers = await client.fetch_trending_papers(
                    queries=self.trending_queries,
                    limit=self.paper_limit,
                    min_citation_count=self.min_citations,
                    max_age_days=self.max_age_days,
                    fields_of_study=["Computer Science"],
                )

            if papers and len(papers) > 0:
                log.info(f"Fetched {len(papers)} papers from Semantic Scholar")
                for i, paper in enumerate(papers.papers[:5], 1):
                    log.debug(
                        f"  [SS-{i}] {paper.title[:60]}... "
                        f"(id={paper.id}, citations={paper.citation_count})"
                    )
                return list(papers.papers)
            else:
                log.warning("No papers fetched from Semantic Scholar")
                return []

        except SemanticScholarAPIError as e:
            log.error(f"Failed to fetch papers from Semantic Scholar: {e}")
            return []
        except Exception as e:
            log.error(f"Unexpected error fetching from Semantic Scholar: {e}")
            return []

    async def collect_papers_once(self) -> int:
        """One-time paper collection (convenience method).

        This is useful for manual collection without starting the scheduler.

        Returns:
            int: Number of papers successfully stored

        Examples:
            >>> count = await scheduler.collect_papers_once()
        """
        return await self.collect_papers()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is currently running.

        Returns:
            bool: True if scheduler is running
        """
        return self._running


async def collect_papers_now(
    vector_store: VectorStoreService,
    paper_limit: int = 30,
) -> int:
    """Convenience function to collect papers immediately.

    This function creates a temporary scheduler and runs a single collection.

    Args:
        vector_store: Vector store service instance
        paper_limit: Maximum papers to fetch

    Returns:
        int: Number of papers successfully stored

    Examples:
        >>> async with VectorStoreService() as vs:
        ...     count = await collect_papers_now(vs, paper_limit=30)
    """
    scheduler = PaperCollectionScheduler(
        vector_store=vector_store,
        paper_limit=paper_limit,
    )
    return await scheduler.collect_papers()
