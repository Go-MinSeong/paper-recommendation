"""Manual paper collection script.

Run this script to manually collect papers from Hugging Face
and store them in Milvus vector database.

Usage:
    cd paper-recommendation
    uv run python scripts/collect_papers.py
    
    # With custom limit
    uv run python scripts/collect_papers.py --limit 50
"""

import argparse
import asyncio
import sys

# Add project root to path
sys.path.insert(0, ".")

from config.logger import log
from mcp_servers.vector_store.service import VectorStoreService
from src.scheduler.collector import PaperCollectionScheduler


async def main(paper_limit: int = 30) -> int:
    """Main function to collect papers.

    Args:
        paper_limit: Maximum number of papers to fetch

    Returns:
        int: Number of papers collected
    """
    log.info("=" * 60)
    log.info("Manual Paper Collection Script")
    log.info("=" * 60)

    try:
        # Initialize vector store
        log.info("Initializing Vector Store...")
        vector_store = VectorStoreService()
        await vector_store.initialize()
        log.info("Vector Store initialized")

        # Create scheduler and collect papers
        scheduler = PaperCollectionScheduler(
            vector_store=vector_store,
            paper_limit=paper_limit,
        )

        # Collect papers
        count = await scheduler.collect_papers()

        # Cleanup
        await vector_store.close()

        log.info(f"\n✅ Successfully collected {count} papers!")
        return count

    except Exception as e:
        log.error(f"❌ Paper collection failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect papers from Hugging Face")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of papers to fetch (default: 30)",
    )
    args = parser.parse_args()

    asyncio.run(main(paper_limit=args.limit))
