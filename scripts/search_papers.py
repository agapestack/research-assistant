#!/usr/bin/env python
"""CLI script to search indexed papers."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Search indexed papers")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection stats instead of searching",
    )
    args = parser.parse_args()

    store = VectorStore(host=args.qdrant_host, port=args.qdrant_port)

    if args.stats:
        print(f"Total documents: {store.count()}")
        return

    results = store.search(args.query, k=args.top_k)
    print(f"Found {len(results)} results for: '{args.query}'\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"    Source: {r['metadata']['source']}, Page: {r['metadata']['page']}")
        print(f"    {r['content'][:200]}...")
        print()


if __name__ == "__main__":
    main()
