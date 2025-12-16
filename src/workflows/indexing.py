"""Prefect flows for paper indexing into vector store using ar5iv HTML."""

import sqlite3
import time
from pathlib import Path

from prefect import flow, get_run_logger, task

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "papers.db"


@task
def get_unindexed_papers(limit: int = 100) -> list[dict]:
    """Fetch papers that haven't been indexed yet."""
    logger = get_run_logger()

    if not DB_PATH.exists():
        logger.warning("Database not found")
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        SELECT arxiv_id, title, authors, abstract, published, categories, arxiv_url
        FROM papers
        WHERE is_indexed = 0 AND (html_available IS NULL OR html_available = 1)
        LIMIT ?
    """,
        (limit,),
    )

    papers = []
    for row in cursor.fetchall():
        papers.append(
            {
                "arxiv_id": row[0],
                "title": row[1],
                "authors": row[2],
                "abstract": row[3],
                "published": row[4],
                "categories": row[5],
                "arxiv_url": row[6],
            }
        )

    conn.close()
    logger.info(f"Found {len(papers)} unindexed papers")
    return papers


@task(retries=2, retry_delay_seconds=5)
def fetch_and_chunk_paper(paper: dict) -> dict:
    """Fetch HTML from ar5iv and chunk into documents."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.services.document_loader import chunk_document, extract_text_from_html
    from src.services.html_fetcher import fetch_paper_html

    logger = get_run_logger()
    arxiv_id = paper["arxiv_id"]

    # Fetch HTML
    paper_html = fetch_paper_html(arxiv_id)

    if not paper_html.success:
        logger.warning(f"HTML not available for {arxiv_id}: {paper_html.error}")
        return {
            "arxiv_id": arxiv_id,
            "status": "no_html",
            "chunks": 0,
            "documents": [],
        }

    # Extract text as pages
    pages = extract_text_from_html(paper_html)
    if not pages:
        logger.warning(f"No content extracted from {arxiv_id}")
        return {
            "arxiv_id": arxiv_id,
            "status": "empty",
            "chunks": 0,
            "documents": [],
        }

    # Chunk the content
    extra_metadata = {
        "title": paper_html.title or paper["title"],
        "authors": paper["authors"],
        "published": paper["published"],
        "categories": paper["categories"],
        "arxiv_url": paper["arxiv_url"],
        "arxiv_id": arxiv_id,
        "source_type": "html",
    }

    chunks = chunk_document(
        pages=pages,
        source=f"ar5iv:{arxiv_id}",
        extra_metadata=extra_metadata,
    )

    logger.info(f"Processed {arxiv_id}: {len(chunks)} chunks")
    return {
        "arxiv_id": arxiv_id,
        "status": "success",
        "chunks": len(chunks),
        "documents": chunks,
    }


@task
def index_chunks(chunks: list, batch_size: int = 100) -> int:
    """Index document chunks into vector store."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config import settings
    from src.services import VectorStore

    logger = get_run_logger()

    if not chunks:
        return 0

    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        embedding_model=settings.embedding_model,
    )

    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vector_store.add_documents(batch)
            total_indexed += len(batch)
        except Exception as e:
            logger.error(f"Failed to index batch: {e}")

    logger.info(f"Indexed {total_indexed} chunks")
    return total_indexed


@task
def update_paper_status(results: list[dict]):
    """Update paper status in database after indexing."""
    logger = get_run_logger()

    if not results:
        return

    conn = sqlite3.connect(DB_PATH)

    indexed_ids = []
    no_html_ids = []

    for result in results:
        if result["status"] == "success":
            indexed_ids.append(result["arxiv_id"])
        elif result["status"] == "no_html":
            no_html_ids.append(result["arxiv_id"])

    if indexed_ids:
        conn.executemany(
            "UPDATE papers SET is_indexed = 1, html_available = 1 WHERE arxiv_id = ?",
            [(id,) for id in indexed_ids],
        )

    if no_html_ids:
        conn.executemany(
            "UPDATE papers SET html_available = 0 WHERE arxiv_id = ?", [(id,) for id in no_html_ids]
        )

    conn.commit()
    conn.close()
    logger.info(f"Updated status: {len(indexed_ids)} indexed, {len(no_html_ids)} no HTML")


@flow(name="process-paper", log_prints=True)
def process_paper(paper: dict) -> dict:
    """Process a single paper: fetch HTML, chunk, and prepare for indexing."""
    return fetch_and_chunk_paper(paper)


@flow(name="index-papers", log_prints=True)
def index_papers_flow(
    limit: int = 50,
    batch_size: int = 100,
    delay_between_fetches: float = 1.0,
) -> dict:
    """
    Main indexing flow: Fetch HTML and index papers into vector store.

    Args:
        limit: Maximum papers to process
        batch_size: Chunks per indexing batch
        delay_between_fetches: Delay between ar5iv requests (rate limiting)

    Returns:
        Summary dict with indexing statistics
    """
    logger = get_run_logger()
    logger.info(f"Starting indexing flow: limit={limit}")

    # Get unindexed papers
    papers = get_unindexed_papers(limit)

    if not papers:
        logger.info("No papers to index")
        return {"processed": 0, "indexed": 0, "no_html": 0, "failed": 0}

    # Process papers (with rate limiting for ar5iv)
    results = []
    for i, paper in enumerate(papers):
        if i > 0:
            time.sleep(delay_between_fetches)
        result = fetch_and_chunk_paper(paper)
        results.append(result)

    # Collect all chunks
    all_chunks = []
    successful = 0
    no_html = 0
    failed = 0

    for result in results:
        if result["status"] == "success":
            all_chunks.extend(result["documents"])
            successful += 1
        elif result["status"] == "no_html":
            no_html += 1
        else:
            failed += 1

    # Index all chunks
    indexed_count = 0
    if all_chunks:
        indexed_count = index_chunks(all_chunks, batch_size)

    # Update database status
    update_paper_status(results)

    summary = {
        "processed": len(papers),
        "successful": successful,
        "no_html": no_html,
        "failed": failed,
        "total_chunks": len(all_chunks),
        "indexed_chunks": indexed_count,
    }

    logger.info(f"Indexing complete: {summary}")
    return summary
