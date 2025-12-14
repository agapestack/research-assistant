"""Benchmark different embedding models for retrieval quality."""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import arxiv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import VectorStore, Reranker, get_embedder, list_available_models
from src.services.document_loader import load_paper_from_html
from src.config import settings


@dataclass
class BenchmarkResult:
    model: str
    dimensions: int
    index_time: float
    query_time: float
    avg_top1_score: float
    avg_top5_score: float
    docs_indexed: int


TEST_QUERIES = [
    "What is retrieval augmented generation?",
    "How do large language models handle context?",
    "What are transformer attention mechanisms?",
    "How does chain of thought prompting improve reasoning?",
    "What are the limitations of fine-tuning LLMs?",
]


def index_papers(model_key: str, papers: list[dict], verbose: bool = False) -> tuple[VectorStore, float]:
    """Index papers with a specific embedding model."""
    start = time.time()
    vs = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        embedding_model=model_key,
    )

    for paper in papers:
        docs = load_paper_from_html(
            arxiv_id=paper["arxiv_id"],
            metadata={
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "published": paper["published"],
                "arxiv_url": paper["arxiv_url"],
            }
        )
        if docs:
            vs.add_documents(docs)
            if verbose:
                print(f"  Indexed {paper['arxiv_id']}: {len(docs)} chunks")

    elapsed = time.time() - start
    return vs, elapsed


def evaluate_retrieval(vs: VectorStore, queries: list[str], k: int = 5) -> tuple[float, float, float]:
    """Evaluate retrieval quality on test queries."""
    total_time = 0
    top1_scores = []
    top5_scores = []

    for query in queries:
        start = time.time()
        results = vs.search(query, k=k)
        total_time += time.time() - start

        if results:
            top1_scores.append(results[0]["score"])
            top5_scores.append(sum(r["score"] for r in results[:5]) / min(5, len(results)))

    avg_query_time = total_time / len(queries)
    avg_top1 = sum(top1_scores) / len(top1_scores) if top1_scores else 0
    avg_top5 = sum(top5_scores) / len(top5_scores) if top5_scores else 0

    return avg_query_time, avg_top1, avg_top5


def benchmark_model(model_key: str, papers: list[dict], verbose: bool = False) -> BenchmarkResult:
    """Run full benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_key}")
    print(f"{'='*60}")

    embedder = get_embedder(model_key)
    print(f"Model: {embedder._model_id}")
    print(f"Dimensions: {embedder.dimensions}")

    print("\nIndexing papers...")
    vs, index_time = index_papers(model_key, papers, verbose)
    doc_count = vs.count()
    print(f"Indexed {doc_count} chunks in {index_time:.2f}s")

    print("\nRunning test queries...")
    query_time, avg_top1, avg_top5 = evaluate_retrieval(vs, TEST_QUERIES)
    print(f"Avg query time: {query_time*1000:.1f}ms")
    print(f"Avg top-1 score: {avg_top1:.4f}")
    print(f"Avg top-5 score: {avg_top5:.4f}")

    return BenchmarkResult(
        model=model_key,
        dimensions=embedder.dimensions,
        index_time=index_time,
        query_time=query_time,
        avg_top1_score=avg_top1,
        avg_top5_score=avg_top5,
        docs_indexed=doc_count,
    )


def print_comparison(results: list[BenchmarkResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<12} {'Dims':>6} {'Index(s)':>9} {'Query(ms)':>10} {'Top1':>8} {'Top5':>8}")
    print("-" * 80)

    # Sort by top5 score descending
    for r in sorted(results, key=lambda x: x.avg_top5_score, reverse=True):
        print(f"{r.model:<12} {r.dimensions:>6} {r.index_time:>9.2f} {r.query_time*1000:>10.1f} {r.avg_top1_score:>8.4f} {r.avg_top5_score:>8.4f}")

    print("-" * 80)
    best = max(results, key=lambda x: x.avg_top5_score)
    print(f"\nBest model: {best.model} (top-5 score: {best.avg_top5_score:.4f})")


def fetch_papers(topic: str, max_results: int) -> list[dict]:
    """Fetch papers from arxiv."""
    query = f'ti:"{topic}" OR abs:"{topic}"'
    client = arxiv.Client(page_size=50, delay_seconds=1)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for paper in client.results(search):
        papers.append({
            "arxiv_id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": paper.published.isoformat(),
            "arxiv_url": paper.entry_id,
        })
    return papers


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--models", nargs="+", default=["minilm", "bge-small", "bge-base"],
                       help="Models to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all available models")
    parser.add_argument("--papers", type=int, default=5, help="Number of papers to index")
    parser.add_argument("--topic", default="retrieval augmented generation", help="Topic to search")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available embedding models:")
        for key, model_id in list_available_models().items():
            print(f"  {key:<12} → {model_id}")
        return

    # Use all models if --all flag is set
    if args.all:
        args.models = list(list_available_models().keys())

    # Fetch test papers
    print(f"Fetching {args.papers} papers on '{args.topic}'...")
    papers = fetch_papers(args.topic, args.papers)
    print(f"Found {len(papers)} papers")

    if not papers:
        print("No papers found. Try a different topic.")
        return

    # Benchmark each model
    results = []
    for model_key in args.models:
        try:
            result = benchmark_model(model_key, papers, args.verbose)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_key}: {e}")

    # Print comparison
    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
