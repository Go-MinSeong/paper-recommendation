"""Slack integration module for AIE Insight Bot.

This module provides Slack Bolt integration for handling commands,
events, and messaging through the Slack API.
"""

from src.slack.app import create_slack_app, start_socket_mode

__all__ = ["create_slack_app", "start_socket_mode"]
