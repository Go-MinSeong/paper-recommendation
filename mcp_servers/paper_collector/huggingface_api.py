"""Hugging Face Papers API client using huggingface_hub library.

This module provides interface to fetch weekly trending papers from Hugging Face
using the official huggingface_hub library.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal, Optional

from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger import log
from mcp_servers.paper_collector.models import Paper, PaperCollection


class HuggingFacePapersAPIError(Exception):
    """Exception raised for Hugging Face Papers API errors."""

    pass


class HuggingFacePapersClient:
    """Client for Hugging Face Papers using huggingface_hub library.

    This client fetches weekly trending papers from Hugging Face Daily Papers.

    Attributes:
        api: HuggingFace Hub API instance
    """

    DEFAULT_MIN_UPVOTES = 10

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize Hugging Face Papers client.

        Args:
            token: HuggingFace API token (optional)

        Examples:
            >>> client = HuggingFacePapersClient()
            >>> papers = await client.fetch_weekly_trending(limit=100)
        """
        self.api = HfApi(token=token)
        self._executor = ThreadPoolExecutor(max_workers=3)

    async def __aenter__(self) -> "HuggingFacePapersClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._executor.shutdown(wait=False)

    def _get_current_week(self) -> str:
        """Get current week in ISO format (YYYY-Www).

        Returns:
            str: Week string like "2025-W04"
        """
        now = datetime.now()
        return now.strftime("%G-W%V")

    def _list_daily_papers_sync(
        self,
        week: Optional[str] = None,
        sort: Optional[Literal["publishedAt", "trending"]] = None,
        limit: Optional[int] = None,
    ) -> list:
        """Synchronous wrapper for list_daily_papers."""
        try:
            papers = list(
                self.api.list_daily_papers(
                    week=week,
                    sort=sort,
                    limit=limit,
                )
            )
            return papers
        except Exception as e:
            log.warning(f"Failed to fetch papers: {e}")
            return []

    async def _list_daily_papers(
        self,
        week: Optional[str] = None,
        sort: Optional[Literal["publishedAt", "trending"]] = None,
        limit: Optional[int] = None,
    ) -> list:
        """Async wrapper for list_daily_papers using thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._list_daily_papers_sync(week, sort, limit),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_papers(
        self,
        limit: int = 100,
        min_upvotes: int = DEFAULT_MIN_UPVOTES,
        week: Optional[str] = None,
        **kwargs,  # Accept extra args for backward compatibility
    ) -> PaperCollection:
        """Fetch weekly trending papers from Hugging Face.

        Args:
            limit: Maximum number of papers to return
            min_upvotes: Minimum upvotes required (default: 10)
            week: Week in ISO format (YYYY-Www). Defaults to current week.

        Returns:
            PaperCollection: Collection of trending papers

        Raises:
            HuggingFacePapersAPIError: If API request fails

        Examples:
            >>> async with HuggingFacePapersClient() as client:
            ...     papers = await client.fetch_papers(limit=100, min_upvotes=10)
        """
        try:
            # Use current week if not specified
            target_week = week or self._get_current_week()

            log.info(
                f"Fetching weekly trending papers from HuggingFace "
                f"(week={target_week}, limit={limit}, min_upvotes={min_upvotes})"
            )

            # Fetch trending papers for the week
            raw_papers = await self._list_daily_papers(
                week=target_week,
                sort="trending",
                limit=500,  # Fetch more to filter by upvotes
            )

            log.debug(f"Received {len(raw_papers)} papers from API")

            # Parse and filter papers
            papers: list[Paper] = []
            filtered_count = 0

            for raw_paper in raw_papers:
                try:
                    paper = self._parse_paper(raw_paper)

                    # Filter by upvotes
                    if paper.upvotes < min_upvotes:
                        filtered_count += 1
                        continue

                    papers.append(paper)

                except Exception as e:
                    log.warning(f"Failed to parse paper: {e}")
                    continue

            log.debug(f"Filtered {filtered_count} papers with upvotes < {min_upvotes}")

            # Sort by upvotes descending (already trending sorted, but ensure order)
            papers.sort(key=lambda p: p.upvotes, reverse=True)

            # Limit results
            papers = papers[:limit]

            log.info(
                f"Successfully fetched {len(papers)} trending papers "
                f"(week={target_week}, upvotes >= {min_upvotes})"
            )

            return PaperCollection(
                papers=papers,
                total=len(papers),
                fetched_at=datetime.now(),
            )

        except Exception as e:
            error_msg = f"Failed to fetch papers: {str(e)}"
            log.error(error_msg)
            raise HuggingFacePapersAPIError(error_msg) from e

    async def fetch_weekly_trending(
        self,
        limit: int = 100,
        min_upvotes: int = DEFAULT_MIN_UPVOTES,
        week: Optional[str] = None,
    ) -> PaperCollection:
        """Alias for fetch_papers - fetch weekly trending papers.

        Args:
            limit: Maximum number of papers to return
            min_upvotes: Minimum upvotes required
            week: Week in ISO format (YYYY-Www). Defaults to current week.

        Returns:
            PaperCollection: Collection of trending papers
        """
        return await self.fetch_papers(
            limit=limit,
            min_upvotes=min_upvotes,
            week=week,
        )

    def _parse_paper(self, raw_paper) -> Paper:
        """Parse PaperInfo from huggingface_hub to Paper model.

        Args:
            raw_paper: PaperInfo object from huggingface_hub

        Returns:
            Paper: Parsed paper object
        """
        # Extract paper ID
        paper_id = getattr(raw_paper, "id", "") or ""

        # Build URL
        url = f"https://huggingface.co/papers/{paper_id}"

        # Extract title
        title = getattr(raw_paper, "title", "") or ""

        # Extract abstract/summary
        abstract = getattr(raw_paper, "summary", "") or ""

        # Extract authors
        authors_list = getattr(raw_paper, "authors", []) or []
        if authors_list:
            if isinstance(authors_list[0], str):
                authors = ", ".join(authors_list)
            else:
                authors = ", ".join(
                    [getattr(a, "name", str(a)) for a in authors_list]
                )
        else:
            authors = ""

        # Extract upvotes
        upvotes = getattr(raw_paper, "upvotes", 0) or 0

        # Extract publication date
        published_at = None
        pub_date = getattr(raw_paper, "publishedAt", None)
        if pub_date:
            if isinstance(pub_date, datetime):
                published_at = pub_date
            elif isinstance(pub_date, str):
                try:
                    published_at = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            url=url,
            authors=authors,
            published_at=published_at,
            upvotes=upvotes,
        )


async def fetch_weekly_trending(
    limit: int = 100,
    min_upvotes: int = HuggingFacePapersClient.DEFAULT_MIN_UPVOTES,
    week: Optional[str] = None,
) -> PaperCollection:
    """Convenience function to fetch weekly trending papers.

    Args:
        limit: Maximum number of papers to fetch
        min_upvotes: Minimum upvotes required (default: 10)
        week: Week in ISO format (YYYY-Www). Defaults to current week.

    Returns:
        PaperCollection: Collection of trending papers

    Examples:
        >>> papers = await fetch_weekly_trending(limit=100)
        >>> for paper in papers:
        ...     print(f"{paper.title} ({paper.upvotes} upvotes)")
    """
    async with HuggingFacePapersClient() as client:
        return await client.fetch_papers(
            limit=limit,
            min_upvotes=min_upvotes,
            week=week,
        )


# Backward compatibility alias
fetch_latest_papers = fetch_weekly_trending
