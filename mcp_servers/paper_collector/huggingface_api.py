"""Hugging Face Papers API client.

This module provides interface to fetch papers from Hugging Face Daily Papers API.
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger import log
from mcp_servers.paper_collector.models import Paper, PaperCollection


class HuggingFacePapersAPIError(Exception):
    """Exception raised for Hugging Face Papers API errors."""

    pass


class HuggingFacePapersClient:
    """Client for Hugging Face Daily Papers API.

    This client fetches the latest popular papers from Hugging Face.
    API endpoint: https://huggingface.co/api/daily_papers

    Attributes:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "https://huggingface.co/api",
        timeout: float = 30.0,
    ) -> None:
        """Initialize Hugging Face Papers client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds

        Examples:
            >>> client = HuggingFacePapersClient()
            >>> papers = await client.fetch_papers(limit=10)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "HuggingFacePapersClient":
        """Async context manager entry.

        Returns:
            HuggingFacePapersClient: Self instance
        """
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def fetch_papers(
        self,
        limit: int = 100,
        days: int = 7,
        sort_by_upvotes: bool = True,
    ) -> PaperCollection:
        """Fetch trending papers from Hugging Face within a date range.

        Fetches papers from the last N days and returns top papers by upvotes.

        Args:
            limit: Maximum number of papers to return (1-200)
            days: Number of days to look back (default: 7 for weekly trending)
            sort_by_upvotes: Sort by upvotes descending (default: True)

        Returns:
            PaperCollection: Collection of top trending papers

        Raises:
            HuggingFacePapersAPIError: If API request fails
            ValueError: If limit is out of range

        Examples:
            >>> async with HuggingFacePapersClient() as client:
            ...     papers = await client.fetch_papers(limit=100, days=7)
            ...     print(f"Fetched {len(papers)} trending papers from last 7 days")
        """
        if not 1 <= limit <= 200:
            raise ValueError(f"Limit must be between 1 and 200, got {limit}")

        if not self._client:
            raise HuggingFacePapersAPIError("Client not initialized. Use async context manager.")

        try:
            log.info(f"Fetching papers from Hugging Face API (limit={limit}, days={days})")

            # Fetch more papers to filter by date range
            fetch_limit = min(limit * 3, 300)  # Fetch more to ensure we have enough after filtering
            url = f"{self.base_url}/daily_papers"
            params = {"limit": fetch_limit}

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Calculate date threshold for filtering
            date_threshold = datetime.now() - timedelta(days=days)

            # Parse response and filter by date
            papers: list[Paper] = []
            for item in data:
                try:
                    paper = self._parse_paper(item)

                    # Filter by publication date (within last N days)
                    if paper.published_at and paper.published_at >= date_threshold:
                        papers.append(paper)
                    elif not paper.published_at:
                        # Include papers without publish date (fallback to created_at)
                        papers.append(paper)

                except Exception as e:
                    log.warning(f"Failed to parse paper: {e}")
                    continue

            # Sort by upvotes if requested
            if sort_by_upvotes:
                papers.sort(key=lambda p: p.upvotes, reverse=True)

            # Limit results
            papers = papers[:limit]

            log.info(
                f"Successfully fetched {len(papers)} trending papers "
                f"from last {days} days (sorted by upvotes)"
            )

            return PaperCollection(
                papers=papers,
                total=len(papers),
                fetched_at=datetime.now(),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error occurred: {e.response.status_code}"
            log.error(error_msg)
            raise HuggingFacePapersAPIError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = f"Request error occurred: {str(e)}"
            log.error(error_msg)
            raise HuggingFacePapersAPIError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            log.error(error_msg)
            raise HuggingFacePapersAPIError(error_msg) from e

    def _parse_paper(self, data: dict) -> Paper:
        """Parse paper data from API response.

        Args:
            data: Raw paper data from API

        Returns:
            Paper: Parsed paper object

        Raises:
            KeyError: If required fields are missing
            ValueError: If data validation fails
        """
        # Hugging Face API response structure
        paper_data = data.get("paper", {})

        return Paper(
            id=paper_data.get("id", data.get("id", "")),
            title=paper_data.get("title", ""),
            abstract=paper_data.get("summary", ""),
            url=paper_data.get("url", f"https://huggingface.co/papers/{paper_data.get('id', '')}"),
            authors=", ".join([author.get("name", "") for author in paper_data.get("authors", [])]),
            published_at=self._parse_date(paper_data.get("publishedAt")),
            upvotes=data.get("upvotes", 0),
        )

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime.

        Args:
            date_str: ISO format date string

        Returns:
            Optional[datetime]: Parsed datetime or None if parsing fails
        """
        if not date_str:
            return None

        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None


async def fetch_latest_papers(
    limit: int = 100,
    days: int = 7,
    sort_by_upvotes: bool = True,
) -> PaperCollection:
    """Convenience function to fetch trending papers.

    Args:
        limit: Maximum number of papers to fetch
        days: Number of days to look back
        sort_by_upvotes: Sort by upvotes descending

    Returns:
        PaperCollection: Collection of papers

    Raises:
        HuggingFacePapersAPIError: If fetching fails

    Examples:
        >>> papers = await fetch_latest_papers(limit=100, days=7)
        >>> for paper in papers:
        ...     print(paper.title, paper.upvotes)
    """
    async with HuggingFacePapersClient() as client:
        return await client.fetch_papers(limit=limit, days=days, sort_by_upvotes=sort_by_upvotes)
