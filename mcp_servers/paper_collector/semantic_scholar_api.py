"""Semantic Scholar API client for fetching citation counts.

This module provides interface to fetch paper metadata including citation counts
from Semantic Scholar Academic Graph API.
"""

import re
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.logger import log


class SemanticScholarAPIError(Exception):
    """Exception raised for Semantic Scholar API errors."""

    pass


class SemanticScholarClient:
    """Client for Semantic Scholar Academic Graph API.

    This client fetches paper metadata including citation counts.
    API docs: https://api.semanticscholar.org/api-docs/

    Attributes:
        base_url: Base URL for the API
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        timeout: float = 10.0,
    ) -> None:
        """Initialize Semantic Scholar client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds

        Examples:
            >>> client = SemanticScholarClient()
            >>> citation_count = await client.get_citation_count("arXiv:2301.00001")
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "SemanticScholarClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL.

        Args:
            url: Paper URL (e.g., https://arxiv.org/abs/2301.00001)

        Returns:
            arXiv ID or None if not found
        """
        # Match arXiv ID patterns: 2301.00001, 2301.00001v1, cs.CV/0001001
        patterns = [
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"huggingface\.co/papers/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"(\d{4}\.\d{4,5}(?:v\d+)?)",  # Direct ID
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Get paper metadata by arXiv ID.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00001")

        Returns:
            Paper metadata dict or None if not found
        """
        if not self._client:
            raise SemanticScholarAPIError("Client not initialized. Use async context manager.")

        try:
            # Remove 'v' suffix if present (e.g., 2301.00001v2 -> 2301.00001)
            clean_id = re.sub(r"v\d+$", "", arxiv_id)
            url = f"{self.base_url}/paper/arXiv:{clean_id}"
            params = {"fields": "citationCount,title,year,publicationDate"}

            log.debug(f"Fetching paper from Semantic Scholar: arXiv:{clean_id}")

            response = await self._client.get(url, params=params)

            if response.status_code == 404:
                log.debug(f"Paper not found in Semantic Scholar: arXiv:{clean_id}")
                return None

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            log.warning(f"Semantic Scholar API error: {e}")
            return None

        except Exception as e:
            log.warning(f"Failed to fetch from Semantic Scholar: {e}")
            return None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        reraise=True,
    )
    async def search_paper_by_title(self, title: str) -> Optional[dict]:
        """Search paper by title and return best match.

        Args:
            title: Paper title

        Returns:
            Paper metadata dict or None if not found
        """
        if not self._client:
            raise SemanticScholarAPIError("Client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/paper/search"
            params = {
                "query": title[:200],  # Limit query length
                "fields": "citationCount,title,year,publicationDate",
                "limit": 1,
            }

            log.debug(f"Searching paper in Semantic Scholar: '{title[:50]}...'")

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]

            return None

        except Exception as e:
            log.warning(f"Failed to search in Semantic Scholar: {e}")
            return None

    async def get_citation_count(
        self,
        paper_url: str,
        paper_title: Optional[str] = None,
    ) -> Optional[int]:
        """Get citation count for a paper.

        Tries arXiv ID first, falls back to title search if not found.

        Args:
            paper_url: Paper URL (may contain arXiv ID)
            paper_title: Paper title for fallback search

        Returns:
            Citation count or None if not found

        Examples:
            >>> async with SemanticScholarClient() as client:
            ...     count = await client.get_citation_count(
            ...         "https://arxiv.org/abs/2301.00001",
            ...         "Paper Title"
            ...     )
        """
        # Try arXiv ID first
        arxiv_id = self._extract_arxiv_id(paper_url)
        if arxiv_id:
            paper = await self.get_paper_by_arxiv_id(arxiv_id)
            if paper and "citationCount" in paper:
                log.debug(f"Found citation count via arXiv ID: {paper['citationCount']}")
                return paper["citationCount"]

        # Fallback to title search
        if paper_title:
            paper = await self.search_paper_by_title(paper_title)
            if paper and "citationCount" in paper:
                log.debug(f"Found citation count via title search: {paper['citationCount']}")
                return paper["citationCount"]

        return None


async def get_citation_count(paper_url: str, paper_title: Optional[str] = None) -> Optional[int]:
    """Convenience function to get citation count.

    Args:
        paper_url: Paper URL
        paper_title: Paper title for fallback

    Returns:
        Citation count or None

    Examples:
        >>> count = await get_citation_count(
        ...     "https://arxiv.org/abs/2301.00001",
        ...     "Paper Title"
        ... )
    """
    async with SemanticScholarClient() as client:
        return await client.get_citation_count(paper_url, paper_title)
