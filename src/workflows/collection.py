"""Prefect flows for arXiv paper collection."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

DATA_DIR = Path("data")

SEARCH_QUERIES = [
    'ti:"large language model" OR abs:"large language model"',
    "ti:LLM OR abs:LLM",
    'ti:"retrieval augmented generation" OR abs:"retrieval augmented generation"',
    "ti:RAG OR abs:RAG",
    'ti:"chain of thought" OR abs:"chain of thought"',
    "ti:GPT OR abs:GPT",
    "ti:transformer OR abs:transformer",
]

CS_CATEGORIES = {"cs.AI", "cs.CL", "cs.LG", "cs.IR", "cs.NE"}


@task(
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6),
)
def search_arxiv(query: str, max_results: int, days_back: int) -> list[dict]:
    """Search arXiv for papers matching a query."""
    logger = get_run_logger()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

    full_query = f"({query}) AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.IR)"
    logger.info(f"Searching: {query[:50]}...")

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for paper in client.results(search):
        if paper.published < cutoff_date:
            break
        if not set(paper.categories) & CS_CATEGORIES:
            continue
        papers.append(
            {
                "arxiv_id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat(),
                "categories": paper.categories,
                "arxiv_url": paper.entry_id,
            }
        )

    logger.info(f"Found {len(papers)} papers for query")
    return papers


@task
def deduplicate_papers(all_papers: list[list[dict]]) -> list[dict]:
    """Remove duplicate papers from multiple query results."""
    logger = get_run_logger()
    seen = set()
    unique = []

    for papers in all_papers:
        for paper in papers:
            if paper["arxiv_id"] not in seen:
                seen.add(paper["arxiv_id"])
                unique.append(paper)

    logger.info(f"Deduplicated to {len(unique)} unique papers")
    return unique


@task
def save_to_db(papers: list[dict]) -> int:
    """Save papers to SQLite database."""
    import sqlite3

    logger = get_run_logger()

    db_path = DATA_DIR / "papers.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            published TEXT,
            categories TEXT,
            arxiv_url TEXT,
            collected_at TEXT,
            is_indexed INTEGER DEFAULT 0,
            html_available INTEGER DEFAULT NULL
        )
    """)

    saved = 0
    for paper in papers:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO papers
                (arxiv_id, title, authors, abstract, published, categories, arxiv_url, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    paper["arxiv_id"],
                    paper["title"],
                    ", ".join(paper["authors"]),
                    paper["abstract"],
                    paper["published"],
                    ", ".join(paper["categories"]),
                    paper["arxiv_url"],
                    datetime.now().isoformat(),
                ),
            )
            if conn.total_changes:
                saved += 1
        except Exception as e:
            logger.warning(f"Failed to save {paper['arxiv_id']}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved {saved} new papers to database")
    return saved


@flow(name="collect-single-query", log_prints=True)
def collect_single_query(
    query: str,
    max_results: int = 100,
    days_back: int = 30,
) -> list[dict]:
    """Collect papers for a single search query."""
    return search_arxiv(query, max_results, days_back)


@flow(name="collect-papers", log_prints=True)
def collect_papers_flow(
    queries: list[str] | None = None,
    max_per_query: int = 100,
    days_back: int = 365,
) -> dict:
    """
    Main flow: Collect papers from arXiv matching LLM/RAG queries.

    Args:
        queries: Custom queries (defaults to SEARCH_QUERIES)
        max_per_query: Maximum results per query
        days_back: How many days back to search

    Returns:
        Summary dict with collection statistics
    """
    logger = get_run_logger()
    queries = queries or SEARCH_QUERIES

    logger.info(f"Starting collection: {len(queries)} queries, {days_back} days back")

    # Run searches in parallel (Prefect handles this)
    all_results = []
    for query in queries:
        results = search_arxiv.submit(query, max_per_query, days_back)
        all_results.append(results)

    # Wait for all searches and deduplicate
    all_papers = [r.result() for r in all_results]
    unique_papers = deduplicate_papers(all_papers)

    # Save to database
    saved_count = save_to_db(unique_papers)

    summary = {
        "queries_run": len(queries),
        "total_found": sum(len(p) for p in all_papers),
        "unique_papers": len(unique_papers),
        "saved_to_db": saved_count,
    }

    logger.info(f"Collection complete: {summary}")
    return summary
