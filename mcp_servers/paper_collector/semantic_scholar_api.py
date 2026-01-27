"""Semantic Scholar API client for fetching citation counts and trending papers.

This module provides interface to fetch paper metadata including citation counts
from Semantic Scholar Academic Graph API. Supports both single paper lookup
and bulk search for trending papers.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from config.logger import log
from mcp_servers.paper_collector.models import Paper, PaperCollection


class SemanticScholarAPIError(Exception):
    """Exception raised for Semantic Scholar API errors."""

    pass


class SemanticScholarRateLimitError(SemanticScholarAPIError):
    """Exception raised when rate limit is exceeded (429)."""

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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=3, min=5, max=60),
        retry=retry_if_exception_type(SemanticScholarRateLimitError),
        reraise=True,
    )
    async def search_trending_papers(
        self,
        query: str,
        limit: int = 100,
        min_citation_count: int = 10,
        year_range: Optional[str] = None,
        fields_of_study: Optional[list[str]] = None,
    ) -> list[dict]:
        """Search for trending papers by citation count.

        Args:
            query: Search query (e.g., "machine learning", "transformer")
            limit: Maximum number of papers to return (max 100 per request)
            min_citation_count: Minimum citation count filter
            year_range: Year filter (e.g., "2024-2025", "2024-")
            fields_of_study: List of fields to filter (e.g., ["Computer Science"])

        Returns:
            List of paper metadata dicts sorted by citation count

        Note:
            Semantic Scholar API rate limit: 100 requests per 5 minutes (free tier).
            This method will retry with exponential backoff on 429 errors.

        Examples:
            >>> async with SemanticScholarClient() as client:
            ...     papers = await client.search_trending_papers(
            ...         query="large language model",
            ...         min_citation_count=50,
            ...         year_range="2024-2025"
            ...     )
        """
        if not self._client:
            raise SemanticScholarAPIError("Client not initialized. Use async context manager.")

        try:
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "fields": "paperId,title,abstract,url,authors,year,publicationDate,citationCount,externalIds",
                "limit": min(limit, 100),  # API max is 100
                "minCitationCount": min_citation_count,
            }

            if year_range:
                params["year"] = year_range

            if fields_of_study:
                params["fieldsOfStudy"] = ",".join(fields_of_study)

            log.info(
                f"Searching trending papers: query='{query}', "
                f"min_citations={min_citation_count}, year={year_range}"
            )

            response = await self._client.get(url, params=params)

            # Handle rate limiting specifically
            if response.status_code == 429:
                log.warning(f"Rate limited by Semantic Scholar API for query '{query}', will retry...")
                raise SemanticScholarRateLimitError("Rate limit exceeded (429)")

            response.raise_for_status()

            data = response.json()
            papers = data.get("data", [])

            # Sort by citation count descending
            papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

            log.info(f"Found {len(papers)} papers for query '{query}'")
            return papers

        except SemanticScholarRateLimitError:
            raise  # Let tenacity handle retry

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                log.warning(f"Rate limited by Semantic Scholar API for query '{query}', will retry...")
                raise SemanticScholarRateLimitError("Rate limit exceeded (429)") from e
            log.error(f"Semantic Scholar API error: {e}")
            raise SemanticScholarAPIError(f"API error: {e.response.status_code}") from e

        except Exception as e:
            log.error(f"Failed to search trending papers: {e}")
            raise SemanticScholarAPIError(str(e)) from e

    async def fetch_trending_papers(
        self,
        queries: list[str],
        limit: int = 100,
        min_citation_count: int = 10,
        max_age_days: int = 365,
        fields_of_study: Optional[list[str]] = None,
        delay_between_requests: float = 3.5,
    ) -> PaperCollection:
        """Fetch trending papers from multiple queries and deduplicate.

        Searches multiple queries sequentially (with rate limiting) and
        returns deduplicated papers sorted by citation count.

        Args:
            queries: List of search queries
            limit: Total papers to return after deduplication
            min_citation_count: Minimum citation count
            max_age_days: Maximum paper age in days
            fields_of_study: Fields to filter
            delay_between_requests: Delay between API calls (default: 3.5s for rate limit)

        Note:
            Semantic Scholar API rate limit: 100 requests per 5 minutes (~3s per request).
            Default delay is 3.5s to stay safely under the rate limit.

        Returns:
            PaperCollection with deduplicated trending papers

        Examples:
            >>> async with SemanticScholarClient() as client:
            ...     papers = await client.fetch_trending_papers(
            ...         queries=["transformer", "large language model", "diffusion model"],
            ...         min_citation_count=20,
            ...         max_age_days=365
            ...     )
        """
        if not self._client:
            raise SemanticScholarAPIError("Client not initialized. Use async context manager.")

        # Calculate year range
        current_year = datetime.now().year
        start_year = (datetime.now() - timedelta(days=max_age_days)).year
        year_range = f"{start_year}-{current_year}"

        log.info(
            f"Fetching trending papers: {len(queries)} queries, "
            f"min_citations={min_citation_count}, year_range={year_range}"
        )

        all_papers: dict[str, dict] = {}  # paperId -> paper for deduplication

        for query in queries:
            try:
                papers = await self.search_trending_papers(
                    query=query,
                    limit=100,  # Fetch max per query
                    min_citation_count=min_citation_count,
                    year_range=year_range,
                    fields_of_study=fields_of_study,
                )

                for paper in papers:
                    paper_id = paper.get("paperId")
                    if paper_id and paper_id not in all_papers:
                        all_papers[paper_id] = paper

                log.debug(f"Query '{query}': found {len(papers)} papers, total unique: {len(all_papers)}")

                # Rate limiting
                await asyncio.sleep(delay_between_requests)

            except SemanticScholarRateLimitError:
                log.warning(f"Rate limit exceeded for query '{query}', waiting longer...")
                await asyncio.sleep(30)  # Wait 30s before next query on rate limit
                continue

            except Exception as e:
                log.warning(f"Failed to fetch papers for query '{query}': {e}")
                continue

        # Convert to Paper models and sort by citation count
        papers_list: list[Paper] = []
        for paper_data in all_papers.values():
            try:
                paper = self._parse_paper_to_model(paper_data)
                if paper:
                    papers_list.append(paper)
            except Exception as e:
                log.warning(f"Failed to parse paper: {e}")
                continue

        # Sort by citation count descending
        papers_list.sort(key=lambda p: p.citation_count or 0, reverse=True)

        # Limit results
        papers_list = papers_list[:limit]

        log.info(f"Fetched {len(papers_list)} unique trending papers from {len(queries)} queries")

        return PaperCollection(
            papers=papers_list,
            total=len(papers_list),
            fetched_at=datetime.now(),
        )

    def _parse_paper_to_model(self, data: dict) -> Optional[Paper]:
        """Parse Semantic Scholar paper data to Paper model.

        Args:
            data: Raw paper data from API

        Returns:
            Paper model or None if parsing fails
        """
        try:
            paper_id = data.get("paperId", "")

            # Try to get arXiv ID for URL
            external_ids = data.get("externalIds", {}) or {}
            arxiv_id = external_ids.get("ArXiv")

            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
                paper_id = arxiv_id  # Use arXiv ID as paper ID
            else:
                url = data.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

            # Parse authors
            authors_list = data.get("authors", []) or []
            authors = ", ".join([a.get("name", "") for a in authors_list if a.get("name")])

            # Parse publication date
            pub_date = None
            pub_date_str = data.get("publicationDate")
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str)
                except ValueError:
                    pass

            if not pub_date and data.get("year"):
                pub_date = datetime(data["year"], 1, 1)

            return Paper(
                id=paper_id,
                title=data.get("title", ""),
                abstract=data.get("abstract", "") or "",
                url=url,
                authors=authors,
                published_at=pub_date,
                upvotes=0,  # Semantic Scholar doesn't have upvotes
                citation_count=data.get("citationCount"),
            )

        except Exception as e:
            log.warning(f"Failed to parse paper to model: {e}")
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


# Default queries for AI/ML trending papers
DEFAULT_TRENDING_QUERIES = [
    "large language model",
    "transformer neural network",
    "diffusion model",
    "vision language model",
    "multimodal learning",
    "reinforcement learning from human feedback",
    "neural network architecture",
    "deep learning optimization",
    "generative AI",
    "foundation model",
]


async def fetch_trending_papers(
    queries: Optional[list[str]] = None,
    limit: int = 100,
    min_citation_count: int = 10,
    max_age_days: int = 365,
    fields_of_study: Optional[list[str]] = None,
) -> PaperCollection:
    """Convenience function to fetch trending papers.

    Args:
        queries: Search queries (defaults to AI/ML queries)
        limit: Total papers to return
        min_citation_count: Minimum citation count
        max_age_days: Maximum paper age in days
        fields_of_study: Fields to filter (defaults to Computer Science)

    Returns:
        PaperCollection with trending papers

    Examples:
        >>> papers = await fetch_trending_papers(
        ...     min_citation_count=20,
        ...     max_age_days=365
        ... )
        >>> print(f"Found {len(papers)} trending papers")
    """
    if queries is None:
        queries = DEFAULT_TRENDING_QUERIES

    if fields_of_study is None:
        fields_of_study = ["Computer Science"]

    async with SemanticScholarClient(timeout=30.0) as client:
        return await client.fetch_trending_papers(
            queries=queries,
            limit=limit,
            min_citation_count=min_citation_count,
            max_age_days=max_age_days,
            fields_of_study=fields_of_study,
        )
