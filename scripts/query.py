#!/usr/bin/env python
"""CLI script to query the RAG chain."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import RAGChain
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Query the RAG chain")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument(
        "--model",
        type=str,
        default=settings.llm_model,
        choices=list(RAGChain.AVAILABLE_MODELS.keys()),
        help=f"LLM model to use (default: {settings.llm_model})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=settings.retrieval_k,
        help=f"Number of documents to retrieve (default: {settings.retrieval_k})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model, desc in RAGChain.AVAILABLE_MODELS.items():
            print(f"  {model}: {desc}")
        return

    print(f"Using model: {args.model}")
    print(f"Question: {args.question}\n")

    rag = RAGChain(
        model=args.model,
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
    )

    result = rag.query(args.question, k=args.k)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    for i, src in enumerate(result["sources"], 1):
        print(f"[{i}] {src['title']}")
        if src["arxiv_url"]:
            print(f"    {src['arxiv_url']}")
        print(f"    Page: {src['page']}, Score: {src['score']:.4f}")
        print()


if __name__ == "__main__":
    main()
