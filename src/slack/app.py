"""Slack Bolt App creation and Socket Mode management.

This module provides the main Slack App instance with Socket Mode
support for real-time command handling without requiring a public URL.
"""

import asyncio
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from config.settings import get_settings
from config.logger import log
from src.recommender.engine import RecommendationEngine
from mcp_servers.interest_manager.storage import InterestStorage
from mcp_servers.recommendation_history.storage import RecommendationHistoryStorage
from mcp_servers.auto_recommend.storage import AutoRecommendStorage


class SlackAppDependencies:
    """Container for Slack App dependencies.

    Attributes:
        recommendation_engine: Engine for generating paper recommendations
        interest_storage: Storage for user interests
        recommendation_history: Storage for recommendation history
        auto_recommend_storage: Storage for auto-recommend settings
        settings: Application settings
    """

    def __init__(
        self,
        recommendation_engine: RecommendationEngine,
        interest_storage: InterestStorage,
        recommendation_history: RecommendationHistoryStorage,
        auto_recommend_storage: AutoRecommendStorage | None = None,
    ):
        """Initialize dependencies container.

        Args:
            recommendation_engine: Engine for generating paper recommendations
            interest_storage: Storage for user interests
            recommendation_history: Storage for recommendation history
            auto_recommend_storage: Storage for auto-recommend settings
        """
        self.recommendation_engine = recommendation_engine
        self.interest_storage = interest_storage
        self.recommendation_history = recommendation_history
        self.auto_recommend_storage = auto_recommend_storage or AutoRecommendStorage()
        self.settings = get_settings()


def create_slack_app(
    recommendation_engine: RecommendationEngine,
    interest_storage: InterestStorage,
    recommendation_history: RecommendationHistoryStorage | None = None,
    auto_recommend_storage: AutoRecommendStorage | None = None,
) -> AsyncApp:
    """Create and configure Slack Bolt AsyncApp with command handlers.

    Args:
        recommendation_engine: Engine for generating paper recommendations
        interest_storage: Storage for user interests
        recommendation_history: Storage for recommendation history (optional, uses engine's if not provided)
        auto_recommend_storage: Storage for auto-recommend settings (optional)

    Returns:
        Configured AsyncApp instance with registered handlers

    Example:
        >>> engine = RecommendationEngine(vector_store, summarizer, history)
        >>> storage = InterestStorage()
        >>> app = create_slack_app(engine, storage)
    """
    settings = get_settings()

    # Create Slack App instance
    app = AsyncApp(
        token=settings.slack_bot_token,
        # Socket Mode doesn't need signing secret for verification
        # But we keep it for future HTTP Mode support
    )

    # Use recommendation_history from engine if not provided
    if recommendation_history is None:
        recommendation_history = recommendation_engine.recommendation_history

    # Create dependencies container
    deps = SlackAppDependencies(
        recommendation_engine=recommendation_engine,
        interest_storage=interest_storage,
        recommendation_history=recommendation_history,
        auto_recommend_storage=auto_recommend_storage,
    )

    # Register command handlers
    _register_command_handlers(app, deps)

    log.info("Slack App created successfully")
    return app


def _register_command_handlers(app: AsyncApp, deps: SlackAppDependencies) -> None:
    """Register Slack command handlers with dependency injection.

    Args:
        app: AsyncApp instance to register handlers to
        deps: Dependencies container with services
    """
    from src.slack.handlers.commands import (
        create_set_interest_handler,
        create_insight_handler,
        create_my_interest_handler,
        create_clear_interest_handler,
        create_history_handler,
        create_auto_recommend_handler,
    )

    # Register /set_interest command
    set_interest_handler = create_set_interest_handler(deps.interest_storage)
    app.command("/set_interest")(set_interest_handler)

    # Register /my_interest command
    my_interest_handler = create_my_interest_handler(deps.interest_storage)
    app.command("/my_interest")(my_interest_handler)

    # Register /clear_interest command
    clear_interest_handler = create_clear_interest_handler(deps.interest_storage)
    app.command("/clear_interest")(clear_interest_handler)

    # Register /insight command
    insight_handler = create_insight_handler(
        deps.recommendation_engine,
        deps.interest_storage,
        deps.settings,
    )
    app.command("/insight")(insight_handler)

    # Register /history command
    history_handler = create_history_handler(deps.recommendation_history)
    app.command("/history")(history_handler)

    # Register /auto_recommend command
    auto_recommend_handler = create_auto_recommend_handler(deps.auto_recommend_storage)
    app.command("/auto_recommend")(auto_recommend_handler)

    log.info(
        "Command handlers registered: "
        "/set_interest, /my_interest, /clear_interest, /insight, /history, /auto_recommend"
    )


async def start_socket_mode(app: AsyncApp) -> None:
    """Start Socket Mode handler for real-time Slack communication.

    This function starts an AsyncSocketModeHandler that maintains a
    WebSocket connection to Slack. It will run indefinitely until
    cancelled or an error occurs.

    Args:
        app: Configured AsyncApp instance

    Raises:
        Exception: If Socket Mode handler fails to start or encounters
            a fatal error during operation

    Example:
        >>> app = create_slack_app(engine, storage)
        >>> await start_socket_mode(app)
    """
    settings = get_settings()

    try:
        log.info("Starting Slack Socket Mode handler...")

        handler = AsyncSocketModeHandler(
            app=app,
            app_token=settings.slack_app_token,
        )

        # Start handler (runs until cancelled)
        await handler.start_async()

    except asyncio.CancelledError:
        log.info("Socket Mode handler cancelled")
        raise
    except Exception as e:
        log.error(f"Socket Mode handler failed: {e}", exc_info=True)
        raise
