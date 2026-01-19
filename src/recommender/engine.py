"""Recommendation engine for personalized paper suggestions.

This module provides the core recommendation logic combining vector search and summarization.
"""

from typing import Any

from pydantic import BaseModel, Field

from config.logger import log
from config.settings import get_settings
from mcp_servers.interest_manager.models import UserInterest
from mcp_servers.vector_store.service import VectorStoreService
from src.recommender.summarizer import PaperSummarizer


class Recommendation(BaseModel):
    """Paper recommendation with summaries.

    Attributes:
        paper_id: Paper ID
        title: Paper title
        abstract: Paper abstract
        url: Paper URL
        upvotes: Number of upvotes
        similarity_score: Cosine similarity score
        core_summary: General summary
        contextualized_summary: Interest-based summary
    """

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    url: str = Field(..., description="Paper URL")
    upvotes: int = Field(..., description="Number of upvotes")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    core_summary: str = Field(..., description="Core summary")
    contextualized_summary: str = Field(..., description="Contextualized summary")


class RecommendationEngine:
    """Engine for generating personalized paper recommendations.

    This engine:
    1. Searches for similar papers based on user interest
    2. Generates core and contextualized summaries
    3. Returns ranked recommendations

    Attributes:
        vector_store: Vector store service for similarity search
        summarizer: Paper summarizer for generating summaries
        top_k: Number of recommendations to generate
        min_score: Minimum similarity score threshold
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        summarizer: PaperSummarizer,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        """Initialize recommendation engine.

        Args:
            vector_store: Vector store service instance
            summarizer: Paper summarizer instance
            top_k: Number of recommendations (uses settings if not provided)
            min_score: Min similarity score (uses settings if not provided)

        Examples:
            >>> vector_store = VectorStoreService()
            >>> summarizer = PaperSummarizer()
            >>> engine = RecommendationEngine(vector_store, summarizer)
            >>> recs = await engine.recommend(user_interest)
        """
        settings = get_settings()

        self.vector_store = vector_store
        self.summarizer = summarizer
        self.top_k = top_k or settings.top_k_recommendations
        self.min_score = min_score or settings.min_similarity_score

        log.info(
            f"Initialized RecommendationEngine (top_k={self.top_k}, min_score={self.min_score})"
        )

    async def recommend(self, user_interest: UserInterest) -> list[Recommendation]:
        """Generate personalized paper recommendations.

        Args:
            user_interest: User interest object

        Returns:
            list[Recommendation]: List of recommendations with summaries

        Raises:
            Exception: If recommendation generation fails

        Examples:
            >>> engine = RecommendationEngine(vector_store, summarizer)
            >>> interest = UserInterest(user_id="U123", interest="VLM research")
            >>> recommendations = await engine.recommend(interest)
            >>> for rec in recommendations:
            ...     print(rec.title, rec.similarity_score)
        """
        try:
            log.info(
                f"Generating recommendations for user {user_interest.user_id}: '{user_interest.interest[:50]}...'"
            )
            log.debug(f"[Engine] User ID: {user_interest.user_id}")
            log.debug(f"[Engine] Full interest text: '{user_interest.interest}'")
            log.debug(f"[Engine] Config: top_k={self.top_k}, min_score={self.min_score}")

            # Search for similar papers
            log.debug("[Engine] Step 1: Searching for similar papers in vector store...")
            similar_papers = await self.vector_store.search_similar_papers(
                query_text=user_interest.interest,
                top_k=self.top_k,
                min_score=self.min_score,
            )

            log.debug(f"[Engine] Vector search returned {len(similar_papers)} papers")

            if not similar_papers:
                log.warning("No papers found matching the criteria")
                log.debug(
                    f"[Engine] Possible reasons: "
                    f"1) No papers in DB, "
                    f"2) No papers above min_score={self.min_score}, "
                    f"3) Interest too specific"
                )
                return []

            # Generate summaries for each paper
            log.debug("[Engine] Step 2: Generating summaries for matched papers...")
            recommendations: list[Recommendation] = []

            for idx, paper in enumerate(similar_papers):
                try:
                    log.debug(
                        f"[Engine] Processing paper {idx+1}/{len(similar_papers)}: "
                        f"'{paper['title'][:50]}...' (score={paper['score']:.4f})"
                    )

                    # Generate core summary
                    log.debug(f"[Engine] Generating core summary for paper {idx+1}...")
                    core_summary = await self.summarizer.generate_core_summary(
                        title=paper["title"],
                        abstract=paper["abstract"],
                    )

                    # Generate contextualized summary
                    log.debug(f"[Engine] Generating contextualized summary for paper {idx+1}...")
                    contextualized_summary = (
                        await self.summarizer.generate_contextualized_summary(
                            title=paper["title"],
                            abstract=paper["abstract"],
                            user_interest=user_interest.interest,
                        )
                    )

                    # Create recommendation
                    recommendation = Recommendation(
                        paper_id=paper["paper_id"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        url=paper["url"],
                        upvotes=paper["upvotes"],
                        similarity_score=paper["score"],
                        core_summary=core_summary,
                        contextualized_summary=contextualized_summary,
                    )

                    recommendations.append(recommendation)

                    log.debug(
                        f"[Engine] Successfully created recommendation for paper {idx+1}: "
                        f"'{paper['title'][:50]}...'"
                    )

                except Exception as e:
                    log.error(f"Failed to generate summaries for paper {idx+1}: {e}")
                    continue

            log.info(f"Generated {len(recommendations)} recommendations")
            log.debug(f"[Engine] Recommendation generation complete. Returning {len(recommendations)} results.")

            return recommendations

        except Exception as e:
            log.error(f"Failed to generate recommendations: {e}")
            raise

    async def recommend_for_team(
        self,
        team_interests: list[UserInterest],
    ) -> dict[str, list[Recommendation]]:
        """Generate recommendations for multiple team members.

        Args:
            team_interests: List of team member interests

        Returns:
            dict[str, list[Recommendation]]: Recommendations per user

        Raises:
            Exception: If recommendation generation fails

        Examples:
            >>> engine = RecommendationEngine(vector_store, summarizer)
            >>> interests = [
            ...     UserInterest(user_id="U123", interest="VLM"),
            ...     UserInterest(user_id="U456", interest="Object detection")
            ... ]
            >>> team_recs = await engine.recommend_for_team(interests)
            >>> len(team_recs)
            2
        """
        try:
            log.info(f"Generating recommendations for {len(team_interests)} team members")

            recommendations_per_user: dict[str, list[Recommendation]] = {}

            for interest in team_interests:
                try:
                    recommendations = await self.recommend(interest)
                    recommendations_per_user[interest.user_id] = recommendations

                except Exception as e:
                    log.error(
                        f"Failed to generate recommendations for user {interest.user_id}: {e}"
                    )
                    recommendations_per_user[interest.user_id] = []

            log.info(
                f"Generated recommendations for {len(recommendations_per_user)} users"
            )

            return recommendations_per_user

        except Exception as e:
            log.error(f"Failed to generate team recommendations: {e}")
            raise
