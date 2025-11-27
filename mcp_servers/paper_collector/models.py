"""Data models for paper collection.

This module defines Pydantic models for paper data structures.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class Paper(BaseModel):
    """Paper data model with validation.

    Attributes:
        id: Unique identifier for the paper
        title: Paper title
        abstract: Paper abstract/summary
        url: Direct link to the paper
        authors: Comma-separated list of authors
        published_at: Publication date
        upvotes: Number of upvotes (popularity metric)
        created_at: Timestamp when paper was collected
    """

    id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., min_length=1, description="Paper title")
    abstract: str = Field(..., min_length=1, description="Paper abstract")
    url: HttpUrl = Field(..., description="Paper URL")
    authors: Optional[str] = Field(default=None, description="Paper authors")
    published_at: Optional[datetime] = Field(
        default=None,
        description="Publication date",
    )
    upvotes: int = Field(default=0, ge=0, description="Number of upvotes")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Collection timestamp",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: lambda v: str(v),
        }

    def to_text(self) -> str:
        """Convert paper to text representation for embedding.

        Returns:
            str: Concatenated title and abstract

        Examples:
            >>> paper = Paper(id="1", title="Test", abstract="Abstract", url="https://example.com")
            >>> paper.to_text()
            'Test\\nAbstract'
        """
        return f"{self.title}\n{self.abstract}"


class PaperCollection(BaseModel):
    """Collection of papers with metadata.

    Attributes:
        papers: List of Paper objects
        total: Total number of papers
        fetched_at: Timestamp when papers were fetched
    """

    papers: list[Paper] = Field(default_factory=list, description="List of papers")
    total: int = Field(default=0, ge=0, description="Total number of papers")
    fetched_at: datetime = Field(
        default_factory=datetime.now,
        description="Fetch timestamp",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def __len__(self) -> int:
        """Get number of papers in collection.

        Returns:
            int: Number of papers
        """
        return len(self.papers)

    def __iter__(self):
        """Iterate over papers.

        Yields:
            Paper: Individual paper objects
        """
        return iter(self.papers)
