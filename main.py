"""AIE Insight Bot - Paper Recommendation System.

This module serves as the main entry point for the application.
It initializes FastAPI server, Slack bot, and scheduler.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.logger import log
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager.

    Handles startup and shutdown events for the application.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    log.info("Starting AIE Insight Bot...")

    settings = get_settings()
    log.info(f"Environment: {settings.environment}")
    log.info(f"Milvus: {settings.milvus_uri}")

    # Initialize services
    from mcp_servers.vector_store.service import VectorStoreService
    from src.recommender.summarizer import PaperSummarizer
    from src.recommender.engine import RecommendationEngine
    from mcp_servers.interest_manager.storage import InterestStorage
    from mcp_servers.recommendation_history.storage import RecommendationHistoryStorage
    from src.slack.app import create_slack_app, start_socket_mode
    from src.scheduler import PaperCollectionScheduler
    from src.scheduler.collector import PaperSource
    from src.scheduler.recommendation import RecommendationScheduler
    from src.scheduler.auto_recommend import AutoRecommendScheduler
    from mcp_servers.auto_recommend.storage import AutoRecommendStorage
    from slack_sdk.web.async_client import AsyncWebClient

    # 1. Initialize Vector Store
    log.info("Initializing Vector Store...")
    vector_store = VectorStoreService()
    await vector_store.initialize()
    log.info("Vector Store initialized")

    # 2. Initialize Summarizer
    log.info("Initializing Paper Summarizer...")
    summarizer = PaperSummarizer()
    log.info("Paper Summarizer initialized")

    # 3. Initialize Interest Storage
    log.info("Initializing Interest Storage...")
    storage = InterestStorage()
    log.info("Interest Storage initialized")

    # 4. Initialize Recommendation History Storage
    log.info("Initializing Recommendation History Storage...")
    recommendation_history = RecommendationHistoryStorage()
    log.info("Recommendation History Storage initialized")

    # 4.5 Initialize Auto-Recommend Storage
    log.info("Initializing Auto-Recommend Storage...")
    auto_recommend_storage = AutoRecommendStorage()
    log.info("Auto-Recommend Storage initialized")

    # 5. Initialize Recommendation Engine
    log.info("Initializing Recommendation Engine...")
    engine = RecommendationEngine(
        vector_store=vector_store,
        summarizer=summarizer,
        recommendation_history=recommendation_history,
        top_k=settings.top_k_recommendations,
        min_score=settings.min_similarity_score,
    )
    log.info("Recommendation Engine initialized")

    # 6. Initialize and Start Paper Collection Scheduler
    log.info("Initializing Paper Collection Scheduler...")
    paper_source = PaperSource(settings.paper_source)
    scheduler = PaperCollectionScheduler(
        vector_store=vector_store,
        interval_hours=settings.collection_interval_hours,
        paper_limit=settings.paper_collection_limit,
        min_upvotes=settings.paper_min_upvotes,
        min_citations=settings.paper_min_citations,
        max_age_days=settings.paper_max_age_days,
        source=paper_source,
    )
    log.info(f"Paper Collection Scheduler initialized (source={paper_source.value})")

    # Start scheduler (collect papers immediately on startup)
    log.info("Starting Paper Collection Scheduler...")
    await scheduler.start(run_immediately=True)
    log.info("Paper Collection Scheduler started")

    # 7. Create and start Slack App
    log.info("Creating Slack App...")
    slack_app = create_slack_app(
        recommendation_engine=engine,
        interest_storage=storage,
        auto_recommend_storage=auto_recommend_storage,
    )
    log.info("Slack App created")

    # Start Socket Mode handler in background
    log.info("Starting Slack Socket Mode handler...")
    handler_task = asyncio.create_task(start_socket_mode(slack_app))
    log.info("Slack Socket Mode handler started")

    # 8. Initialize and Start Recommendation Scheduler (if enabled)
    recommendation_scheduler = None
    if settings.auto_recommend_enabled:
        log.info("Initializing Recommendation Scheduler...")
        slack_client = AsyncWebClient(token=settings.slack_bot_token)
        recommendation_scheduler = RecommendationScheduler(
            engine=engine,
            interest_storage=storage,
            slack_client=slack_client,
            interval_hours=settings.auto_recommend_interval_hours,
        )
        log.info("Starting Recommendation Scheduler...")
        await recommendation_scheduler.start(run_immediately=False)
        log.info("Recommendation Scheduler started")
    else:
        log.info("Recommendation Scheduler is disabled (AUTO_RECOMMEND_ENABLED=false)")

    # 9. Initialize and Start User Auto-Recommend Scheduler
    log.info("Initializing User Auto-Recommend Scheduler...")
    slack_client = AsyncWebClient(token=settings.slack_bot_token)
    auto_recommend_scheduler = AutoRecommendScheduler(
        auto_recommend_storage=auto_recommend_storage,
        interest_storage=storage,
        recommendation_engine=engine,
        slack_client=slack_client,
        check_interval=60.0,  # Check every minute
    )
    await auto_recommend_scheduler.start()
    log.info("User Auto-Recommend Scheduler started")

    log.info("AIE Insight Bot started successfully")

    yield

    # Shutdown
    log.info("Shutting down AIE Insight Bot...")

    # Stop Recommendation Scheduler (if running)
    if recommendation_scheduler and recommendation_scheduler.is_running:
        log.info("Stopping Recommendation Scheduler...")
        await recommendation_scheduler.stop()
        log.info("Recommendation Scheduler stopped")

    # Stop User Auto-Recommend Scheduler
    log.info("Stopping User Auto-Recommend Scheduler...")
    await auto_recommend_scheduler.stop()
    log.info("User Auto-Recommend Scheduler stopped")

    # Stop Paper Collection Scheduler
    log.info("Stopping Paper Collection Scheduler...")
    await scheduler.stop()
    log.info("Paper Collection Scheduler stopped")

    # Stop Slack Socket Mode handler
    log.info("Stopping Slack Socket Mode handler...")
    handler_task.cancel()
    try:
        await handler_task
    except asyncio.CancelledError:
        log.info("Slack Socket Mode handler stopped")

    # Close Vector Store connection
    log.info("Closing Vector Store connection...")
    await vector_store.close()
    log.info("Vector Store connection closed")

    # Close Summarizer
    log.info("Closing Paper Summarizer...")
    await summarizer.close()
    log.info("Paper Summarizer closed")

    log.info("AIE Insight Bot shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance

    Examples:
        >>> app = create_app()
        >>> app.title
        'AIE Insight Bot'
    """
    settings = get_settings()

    app = FastAPI(
        title="AIE Insight Bot",
        description="팀 맞춤형 논문/데이터 리서치 자동화 시스템",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware with configurable origins
    cors_origins = settings.cors_origins_list

    # Warn if using wildcard in production
    if settings.is_production() and "*" in cors_origins:
        log.warning(
            "CORS is configured with wildcard '*' in production. "
            "Consider setting CORS_ALLOWED_ORIGINS to specific domains."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True if "*" not in cors_origins else False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict[str, str]: Health status response

    Examples:
        >>> response = await health_check()
        >>> response["status"]
        'healthy'
    """
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        dict[str, str]: Welcome message

    Examples:
        >>> response = await root()
        >>> "AIE Insight Bot" in response["message"]
        True
    """
    return {"message": "Welcome to AIE Insight Bot API"}


@app.get("/info")
async def info() -> dict[str, Any]:
    """Application info endpoint.

    Returns:
        dict[str, Any]: Application information

    Examples:
        >>> response = await info()
        >>> response["name"]
        'AIE Insight Bot'
    """
    settings = get_settings()

    return {
        "name": "AIE Insight Bot",
        "version": "1.0.0",
        "environment": settings.environment,
        "description": "팀 맞춤형 논문/데이터 리서치 자동화 시스템",
    }


if __name__ == "__main__":
    settings = get_settings()

    log.info("Starting server...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development(),
        log_level=settings.log_level.lower(),
    )
