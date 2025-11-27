"""Hugging Face Papers API client.

This module provides interface to fetch papers from Hugging Face Daily Papers API.
"""

from datetime import datetime
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
    async def fetch_papers(self, limit: int = 30) -> PaperCollection:
        """Fetch latest papers from Hugging Face.

        Args:
            limit: Maximum number of papers to fetch (1-100)

        Returns:
            PaperCollection: Collection of papers

        Raises:
            HuggingFacePapersAPIError: If API request fails
            ValueError: If limit is out of range

        Examples:
            >>> async with HuggingFacePapersClient() as client:
            ...     papers = await client.fetch_papers(limit=30)
            ...     print(f"Fetched {len(papers)} papers")
        """
        if not 1 <= limit <= 100:
            raise ValueError(f"Limit must be between 1 and 100, got {limit}")

        if not self._client:
            raise HuggingFacePapersAPIError("Client not initialized. Use async context manager.")

        try:
            log.info(f"Fetching papers from Hugging Face API (limit={limit})")

            url = f"{self.base_url}/daily_papers"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()

            # Parse response and create Paper objects
            papers: list[Paper] = []
            for item in data[:limit]:
                try:
                    paper = self._parse_paper(item)
                    papers.append(paper)
                except Exception as e:
                    log.warning(f"Failed to parse paper: {e}")
                    continue

            log.info(f"Successfully fetched {len(papers)} papers")

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


async def fetch_latest_papers(limit: int = 30) -> PaperCollection:
    """Convenience function to fetch latest papers.

    Args:
        limit: Maximum number of papers to fetch

    Returns:
        PaperCollection: Collection of papers

    Raises:
        HuggingFacePapersAPIError: If fetching fails

    Examples:
        >>> papers = await fetch_latest_papers(limit=30)
        >>> for paper in papers:
        ...     print(paper.title)
    """
    async with HuggingFacePapersClient() as client:
        return await client.fetch_papers(limit=limit)
