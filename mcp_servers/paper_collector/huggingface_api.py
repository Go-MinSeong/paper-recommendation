"""Hugging Face Papers API client using huggingface_hub library.

This module provides interface to fetch weekly trending papers from Hugging Face
using the official huggingface_hub library.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal, Optional

import httpx
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
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "HuggingFacePapersClient":
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self._executor.shutdown(wait=False)
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_current_week(self) -> str:
        """Get current week in ISO format (YYYY-Www).

        Returns:
            str: Week string like "2025-W04"
        """
        now = datetime.now()
        return now.strftime("%G-W%V")

    async def fetch_ai_summary(self, paper_id: str) -> Optional[str]:
        """Fetch AI-generated summary from HuggingFace API.

        Args:
            paper_id: Paper ID (arxiv ID)

        Returns:
            Optional[str]: AI-generated summary or None if not available

        Examples:
            >>> async with HuggingFacePapersClient() as client:
            ...     summary = await client.fetch_ai_summary("2601.16973")
        """
        if not self._http_client:
            log.warning("HTTP client not initialized. Use async context manager.")
            return None

        try:
            url = f"https://huggingface.co/api/papers/{paper_id}"
            response = await self._http_client.get(url)

            if response.status_code == 200:
                data = response.json()
                ai_summary = data.get("ai_summary")
                if ai_summary:
                    log.debug(f"Fetched AI summary for paper {paper_id}")
                    return ai_summary
            else:
                log.debug(f"No AI summary found for paper {paper_id}: HTTP {response.status_code}")

        except Exception as e:
            log.warning(f"Failed to fetch AI summary for {paper_id}: {e}")

        return None

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

            # Parse and filter papers (first pass without AI summary)
            papers: list[Paper] = []
            raw_paper_map: dict[str, object] = {}  # Map paper_id to raw_paper for later
            filtered_count = 0

            for raw_paper in raw_papers:
                try:
                    paper = self._parse_paper(raw_paper)

                    # Filter by upvotes
                    if paper.upvotes < min_upvotes:
                        filtered_count += 1
                        continue

                    papers.append(paper)
                    raw_paper_map[paper.id] = raw_paper

                except Exception as e:
                    log.warning(f"Failed to parse paper: {e}")
                    continue

            log.debug(f"Filtered {filtered_count} papers with upvotes < {min_upvotes}")

            # Sort by upvotes descending (already trending sorted, but ensure order)
            papers.sort(key=lambda p: p.upvotes, reverse=True)

            # Limit results
            papers = papers[:limit]

            # Fetch AI summaries for limited papers
            if papers and self._http_client:
                log.info(f"Fetching AI summaries for {len(papers)} papers...")
                final_papers: list[Paper] = []
                for paper in papers:
                    ai_summary = await self.fetch_ai_summary(paper.id)
                    if ai_summary:
                        # Re-parse with AI summary
                        raw_paper = raw_paper_map.get(paper.id)
                        if raw_paper:
                            paper = self._parse_paper(raw_paper, ai_summary=ai_summary)
                    final_papers.append(paper)
                papers = final_papers

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

    def _parse_paper(self, raw_paper, ai_summary: Optional[str] = None) -> Paper:
        """Parse PaperInfo from huggingface_hub to Paper model.

        Args:
            raw_paper: PaperInfo object from huggingface_hub
            ai_summary: Optional AI-generated summary

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
            ai_summary=ai_summary,
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
