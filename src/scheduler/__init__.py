"""Paper collection scheduler module.

This module provides scheduled tasks for collecting papers from Hugging Face
and storing them in the vector database.
"""

from src.scheduler.collector import PaperCollectionScheduler

__all__ = ["PaperCollectionScheduler"]
