#!/usr/bin/env python
"""CLI script to index papers from arXiv using HTML source."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import VectorStore, load_paper_from_html, fetch_paper_html
from src.workflows.collection import search_arxiv


def main():
    parser = argparse.ArgumentParser(description="Index papers from arXiv (HTML-based)")
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic to search for on arxiv",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max papers to fetch (default: 5)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store before indexing",
    )
    args = parser.parse_args()

    store = VectorStore()

    if args.clear:
        print("Clearing vector store...")
        store.clear()

    print(f"Searching arXiv for: {args.topic}")
    query = f'ti:"{args.topic}" OR abs:"{args.topic}"'
    papers = search_arxiv.fn(query, args.max_results, days_back=365)
    print(f"Found {len(papers)} papers")

    indexed = 0
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        print(f"  Processing {arxiv_id}: {paper['title'][:50]}...")

        docs = load_paper_from_html(
            arxiv_id=arxiv_id,
            metadata={
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "published": paper["published"],
                "categories": ", ".join(paper["categories"]),
                "arxiv_url": paper["arxiv_url"],
            }
        )

        if docs:
            store.add_documents(docs)
            indexed += 1
            print(f"    Indexed {len(docs)} chunks")
        else:
            print(f"    No HTML available, skipping")

    print(f"\nIndexed {indexed}/{len(papers)} papers")
    print(f"Total documents in store: {store.count()}")


if __name__ == "__main__":
    main()
