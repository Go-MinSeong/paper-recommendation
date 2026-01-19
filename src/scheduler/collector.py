"""Paper collection scheduler.

This module handles scheduled collection of papers from Hugging Face
and their storage in Milvus vector database.
"""

import asyncio
from datetime import datetime
from typing import Optional

from config.logger import log
from mcp_servers.paper_collector.huggingface_api import (
    HuggingFacePapersClient,
    HuggingFacePapersAPIError,
)
from mcp_servers.vector_store.service import VectorStoreService


class PaperCollectionScheduler:
    """Scheduler for paper collection tasks.

    This scheduler periodically fetches papers from Hugging Face
    and stores them in the vector database with embeddings.

    Attributes:
        vector_store: Vector store service instance
        interval_hours: Collection interval in hours
        paper_limit: Maximum number of papers to fetch per collection
        _running: Whether the scheduler is currently running
        _task: Background task reference
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        interval_hours: float = 24.0,
        paper_limit: int = 30,
    ) -> None:
        """Initialize paper collection scheduler.

        Args:
            vector_store: Vector store service for storing papers
            interval_hours: Interval between collections (default: 24 hours)
            paper_limit: Maximum papers to fetch per collection (default: 30)

        Examples:
            >>> scheduler = PaperCollectionScheduler(
            ...     vector_store=vector_store,
            ...     interval_hours=24.0,
            ...     paper_limit=30,
            ... )
        """
        self.vector_store = vector_store
        self.interval_hours = interval_hours
        self.paper_limit = paper_limit
        self._running = False
        self._task: Optional[asyncio.Task] = None

        log.info(
            f"Paper collection scheduler initialized: "
            f"interval={interval_hours}h, limit={paper_limit}"
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
        """Collect papers from Hugging Face and store in vector database.

        Returns:
            int: Number of papers successfully stored

        Examples:
            >>> count = await scheduler.collect_papers()
            >>> print(f"Collected {count} papers")
        """
        start_time = datetime.now()
        log.info("=" * 60)
        log.info("Starting paper collection task")
        log.info("=" * 60)

        try:
            # Step 1: Fetch papers from Hugging Face
            log.info(f"Step 1: Fetching papers from Hugging Face (limit={self.paper_limit})")
            async with HuggingFacePapersClient() as client:
                papers = await client.fetch_papers(limit=self.paper_limit)

            if not papers or len(papers) == 0:
                log.warning("No papers fetched from Hugging Face")
                return 0

            log.info(f"Fetched {len(papers)} papers from Hugging Face")

            # Log paper details
            for i, paper in enumerate(papers, 1):
                log.debug(
                    f"  [{i}] {paper.title[:60]}... "
                    f"(id={paper.id}, upvotes={paper.upvotes})"
                )

            # Step 2: Store papers in vector database
            log.info("Step 2: Storing papers in vector database with embeddings")
            inserted_ids = await self.vector_store.store_papers(papers)

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            log.info("=" * 60)
            log.info(f"Paper collection completed in {elapsed:.1f}s")
            log.info(f"Papers fetched: {len(papers)}")
            log.info(f"Papers stored: {len(inserted_ids)}")
            log.info("=" * 60)

            return len(inserted_ids)

        except HuggingFacePapersAPIError as e:
            log.error(f"Failed to fetch papers from Hugging Face: {e}")
            return 0

        except Exception as e:
            log.error(f"Paper collection task failed: {e}")
            return 0

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
