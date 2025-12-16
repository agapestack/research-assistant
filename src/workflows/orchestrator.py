"""Main orchestrator flow combining collection and indexing."""

from prefect import flow, get_run_logger
from prefect.runtime import flow_run

from .collection import collect_papers_flow
from .indexing import index_papers_flow


@flow(name="full-pipeline", log_prints=True)
def full_pipeline_flow(
    days_back: int = 30,
    max_per_query: int = 100,
    index_limit: int = 50,
) -> dict:
    """
    Full pipeline: Collect papers from arXiv and index them.

    Args:
        days_back: How many days back to search
        max_per_query: Maximum results per search query
        index_limit: Maximum papers to index per run
    """
    logger = get_run_logger()
    run_name = flow_run.get_name() if flow_run else "manual"
    logger.info(f"Starting full pipeline: {run_name}")

    logger.info("=" * 50)
    logger.info("PHASE 1: Collecting papers from arXiv")
    logger.info("=" * 50)

    collection_result = collect_papers_flow(
        max_per_query=max_per_query,
        days_back=days_back,
    )

    # Phase 2: Indexing
    logger.info("=" * 50)
    logger.info("PHASE 2: Indexing papers into vector store")
    logger.info("=" * 50)

    indexing_result = index_papers_flow(limit=index_limit)

    # Combined summary
    summary = {
        "collection": collection_result,
        "indexing": indexing_result,
        "status": "success",
    }

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Collected: {collection_result.get('unique_papers', 0)} papers")
    logger.info(
        f"Indexed: {indexing_result.get('successful', 0)} papers ({indexing_result.get('indexed_chunks', 0)} chunks)"
    )
    logger.info("=" * 50)

    return summary


@flow(name="daily-update", log_prints=True)
def daily_update_flow() -> dict:
    """Daily update: Collect recent papers and index them."""
    return full_pipeline_flow(
        days_back=7,
        max_per_query=50,
        index_limit=30,
    )


@flow(name="weekly-full-sync", log_prints=True)
def weekly_full_sync_flow() -> dict:
    """Weekly full sync: Comprehensive collection and indexing."""
    return full_pipeline_flow(
        days_back=30,
        max_per_query=200,
        index_limit=100,
    )
