"""Test script for Hugging Face Papers API.

Run: cd paper-recommendation && uv run python tests/test_hf_api.py
"""

import asyncio

from mcp_servers.paper_collector.huggingface_api import HuggingFacePapersClient


async def test_fetch_papers() -> None:
    """Test fetching papers from Hugging Face API."""
    print("=" * 60)
    print("Testing Hugging Face Papers API")
    print("=" * 60)
    
    async with HuggingFacePapersClient() as client:
        try:
            papers = await client.fetch_papers(limit=5)
            
            print(f"\n✅ Successfully fetched {len(papers)} papers!\n")
            
            for i, paper in enumerate(papers, 1):
                print(f"[{i}] {paper.title}")
                print(f"    ID: {paper.id}")
                print(f"    URL: {paper.url}")
                print(f"    Upvotes: {paper.upvotes}")
                print(f"    Authors: {paper.authors[:50]}..." if paper.authors else "    Authors: N/A")
                print(f"    Abstract: {paper.abstract[:100]}..." if paper.abstract else "    Abstract: N/A")
                print()
            
            return papers
            
        except Exception as e:
            print(f"\n❌ Error fetching papers: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(test_fetch_papers())
