"""Recommendation engine for personalized paper suggestions.

This module provides the core recommendation logic combining vector search and summarization.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from config.logger import log
from config.settings import get_settings
from mcp_servers.interest_manager.models import UserInterest
from mcp_servers.recommendation_history.storage import RecommendationHistoryStorage
from mcp_servers.vector_store.service import VectorStoreService
from mcp_servers.paper_collector.semantic_scholar_api import SemanticScholarClient
from src.recommender.summarizer import PaperSummarizer


class Recommendation(BaseModel):
    """Paper recommendation with summaries.

    Attributes:
        paper_id: Paper ID
        title: Paper title
        abstract: Paper abstract
        url: Paper URL
        published_at: Publication date
        citation_count: Number of citations from Semantic Scholar
        core_summary: General summary
        contextualized_summary: Interest-based summary
    """

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    url: str = Field(..., description="Paper URL")
    published_at: Optional[datetime] = Field(default=None, description="Publication date")
    citation_count: Optional[int] = Field(default=None, description="Citation count")
    core_summary: str = Field(..., description="Core summary")
    contextualized_summary: str = Field(..., description="Contextualized summary")


class RecommendationEngine:
    """Engine for generating personalized paper recommendations.

    This engine:
    1. Searches for similar papers based on user interest
    2. Filters out previously recommended papers
    3. Generates core and contextualized summaries
    4. Records recommendations to history
    5. Returns ranked recommendations

    Attributes:
        vector_store: Vector store service for similarity search
        summarizer: Paper summarizer for generating summaries
        recommendation_history: Storage for tracking recommended papers
        top_k: Number of recommendations to generate
        min_score: Minimum similarity score threshold
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        summarizer: PaperSummarizer,
        recommendation_history: RecommendationHistoryStorage,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        """Initialize recommendation engine.

        Args:
            vector_store: Vector store service instance
            summarizer: Paper summarizer instance
            recommendation_history: Storage for tracking recommended papers
            top_k: Number of recommendations (uses settings if not provided)
            min_score: Min similarity score (uses settings if not provided)

        Examples:
            >>> vector_store = VectorStoreService()
            >>> summarizer = PaperSummarizer()
            >>> history = RecommendationHistoryStorage()
            >>> engine = RecommendationEngine(vector_store, summarizer, history)
            >>> recs = await engine.recommend(user_interest)
        """
        settings = get_settings()

        self.vector_store = vector_store
        self.summarizer = summarizer
        self.recommendation_history = recommendation_history
        self.top_k = top_k or settings.top_k_recommendations
        self.min_score = min_score or settings.min_similarity_score

        log.info(
            f"Initialized RecommendationEngine (top_k={self.top_k}, min_score={self.min_score})"
        )

    async def recommend(self, user_interest: UserInterest) -> list[Recommendation]:
        """Generate personalized paper recommendations.

        Filters out papers that have been previously recommended to any team member.

        Args:
            user_interest: User interest object

        Returns:
            list[Recommendation]: List of recommendations with summaries

        Raises:
            Exception: If recommendation generation fails

        Examples:
            >>> engine = RecommendationEngine(vector_store, summarizer, history)
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

            # Step 1: Load previously recommended paper IDs
            log.debug("[Engine] Step 1: Loading recommendation history...")
            recommended_ids = await self.recommendation_history.get_recommended_ids()
            log.debug(f"[Engine] Found {len(recommended_ids)} previously recommended papers")

            # Step 2: Search for similar papers (fetch more to account for filtering)
            search_limit = self.top_k + len(recommended_ids) if recommended_ids else self.top_k
            search_limit = min(search_limit, self.top_k * 3)  # Cap at 3x to avoid excessive search

            log.debug(f"[Engine] Step 2: Searching for similar papers (limit={search_limit})...")
            similar_papers = await self.vector_store.search_similar_papers(
                query_text=user_interest.interest,
                top_k=search_limit,
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

            # Step 3: Filter out previously recommended papers
            log.debug("[Engine] Step 3: Filtering out previously recommended papers...")
            new_papers = [p for p in similar_papers if p["paper_id"] not in recommended_ids]
            filtered_count = len(similar_papers) - len(new_papers)

            if filtered_count > 0:
                log.info(f"Filtered out {filtered_count} previously recommended papers")

            if not new_papers:
                log.warning("All matching papers have been previously recommended")
                return []

            # Limit to top_k after filtering
            new_papers = new_papers[: self.top_k]

            # Step 4: Generate summaries and fetch metadata for each paper
            log.debug(f"[Engine] Step 4: Generating summaries for {len(new_papers)} papers...")
            recommendations: list[Recommendation] = []

            async with SemanticScholarClient() as ss_client:
                for idx, paper in enumerate(new_papers):
                    try:
                        log.debug(
                            f"[Engine] Processing paper {idx+1}/{len(new_papers)}: "
                            f"'{paper['title'][:50]}...' (score={paper['score']:.4f})"
                        )

                        # Fetch citation count and publication date from Semantic Scholar
                        log.debug(f"[Engine] Fetching metadata from Semantic Scholar for paper {idx+1}...")
                        citation_count = None
                        published_at = None

                        try:
                            ss_paper = await ss_client.get_paper_by_arxiv_id(paper["paper_id"])
                            if not ss_paper:
                                ss_paper = await ss_client.search_paper_by_title(paper["title"])

                            if ss_paper:
                                citation_count = ss_paper.get("citationCount")
                                pub_date_str = ss_paper.get("publicationDate")
                                if pub_date_str:
                                    try:
                                        published_at = datetime.fromisoformat(pub_date_str)
                                    except ValueError:
                                        pass
                                log.debug(
                                    f"[Engine] Semantic Scholar: citations={citation_count}, "
                                    f"published={pub_date_str}"
                                )
                        except Exception as e:
                            log.warning(f"Failed to fetch Semantic Scholar data: {e}")

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
                            published_at=published_at,
                            citation_count=citation_count,
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

            # Step 5: Record recommendations to history
            log.debug("[Engine] Step 5: Recording recommendations to history...")
            for rec in recommendations:
                await self.recommendation_history.add_recommendation(
                    paper_id=rec.paper_id,
                    title=rec.title,
                    user_id=user_interest.user_id,
                )

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
