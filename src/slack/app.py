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


class SlackAppDependencies:
    """Container for Slack App dependencies.

    Attributes:
        recommendation_engine: Engine for generating paper recommendations
        interest_storage: Storage for user interests
        settings: Application settings
    """

    def __init__(
        self,
        recommendation_engine: RecommendationEngine,
        interest_storage: InterestStorage,
    ):
        """Initialize dependencies container.

        Args:
            recommendation_engine: Engine for generating paper recommendations
            interest_storage: Storage for user interests
        """
        self.recommendation_engine = recommendation_engine
        self.interest_storage = interest_storage
        self.settings = get_settings()


def create_slack_app(
    recommendation_engine: RecommendationEngine,
    interest_storage: InterestStorage,
) -> AsyncApp:
    """Create and configure Slack Bolt AsyncApp with command handlers.

    Args:
        recommendation_engine: Engine for generating paper recommendations
        interest_storage: Storage for user interests

    Returns:
        Configured AsyncApp instance with registered handlers

    Example:
        >>> engine = RecommendationEngine(vector_store, summarizer)
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

    # Create dependencies container
    deps = SlackAppDependencies(
        recommendation_engine=recommendation_engine,
        interest_storage=interest_storage,
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
    )

    # Register /set_interest command
    set_interest_handler = create_set_interest_handler(deps.interest_storage)
    app.command("/set_interest")(set_interest_handler)

    # Register /insight command
    insight_handler = create_insight_handler(
        deps.recommendation_engine,
        deps.interest_storage,
        deps.settings,
    )
    app.command("/insight")(insight_handler)

    log.info("Command handlers registered: /set_interest, /insight")


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
