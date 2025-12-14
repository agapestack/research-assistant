"""Evaluate retrieval quality with test queries."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import VectorStore, Reranker
from src.config import settings


TEST_QUERIES = [
    "What is retrieval augmented generation?",
    "How do transformers handle long sequences?",
    "What are the limitations of large language models?",
    "How does chain of thought prompting work?",
    "What is the difference between fine-tuning and prompting?",
]


def evaluate_retrieval(k: int = 5, use_reranker: bool = True):
    vs = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        embedding_model=settings.embedding_model,
    )
    reranker = Reranker() if use_reranker else None

    print(f"Collection: {vs.collection_name}")
    print(f"Documents: {vs.count()}")
    print(f"Reranker: {'enabled' if use_reranker else 'disabled'}")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Fetch more if reranking
        fetch_k = 20 if use_reranker else k
        results = vs.search(query, k=fetch_k)

        if use_reranker and reranker:
            results = reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]

        if not results:
            print("  No results found")
            continue

        for i, r in enumerate(results, 1):
            title = r["metadata"].get("title", "Unknown")[:50]
            score = r.get("rerank_score", r.get("score", 0))
            orig = r.get("original_score", r.get("score", 0))
            content_preview = r["content"][:100].replace("\n", " ")

            print(f"  [{i}] {title}...")
            print(f"      Score: {score:.4f} (orig: {orig:.4f})")
            print(f"      Content: {content_preview}...")
        print()


def compare_models():
    """Compare retrieval with and without reranker."""
    print("=" * 60)
    print("WITHOUT RERANKER")
    print("=" * 60)
    evaluate_retrieval(k=5, use_reranker=False)

    print("\n" + "=" * 60)
    print("WITH RERANKER")
    print("=" * 60)
    evaluate_retrieval(k=5, use_reranker=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare with/without reranker")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate_retrieval(k=args.k, use_reranker=not args.no_rerank)
