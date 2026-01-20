"""Models for recommendation history management.

This module provides data models for tracking recommended papers.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class RecommendedPaper(BaseModel):
    """A paper that has been recommended.

    Attributes:
        paper_id: Unique identifier for the paper
        title: Paper title
        recommended_at: Timestamp when the paper was recommended
        recommended_to_user_id: Slack user ID who received the recommendation
    """

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    recommended_at: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    recommended_to_user_id: str = Field(..., description="User ID who received the recommendation")


class RecommendationHistory(BaseModel):
    """History of all recommended papers for the team.

    This model tracks all papers that have been recommended to any team member
    to prevent duplicate recommendations.

    Attributes:
        recommended_paper_ids: Set of paper IDs that have been recommended
        details: List of recommendation details
    """

    recommended_paper_ids: set[str] = Field(default_factory=set, description="Set of recommended paper IDs")
    details: list[RecommendedPaper] = Field(default_factory=list, description="Recommendation details")

    def add(self, paper_id: str, title: str, user_id: str) -> None:
        """Add a paper to recommendation history.

        Args:
            paper_id: Paper ID
            title: Paper title
            user_id: User ID who received the recommendation
        """
        self.recommended_paper_ids.add(paper_id)
        self.details.append(
            RecommendedPaper(
                paper_id=paper_id,
                title=title,
                recommended_to_user_id=user_id,
            )
        )

    def is_recommended(self, paper_id: str) -> bool:
        """Check if a paper has been recommended.

        Args:
            paper_id: Paper ID to check

        Returns:
            bool: True if the paper has been recommended
        """
        return paper_id in self.recommended_paper_ids

    def __len__(self) -> int:
        """Return the number of recommended papers."""
        return len(self.recommended_paper_ids)
