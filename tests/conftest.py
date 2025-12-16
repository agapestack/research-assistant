"""Shared pytest fixtures."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CACHED_PAPERS_FILE = FIXTURES_DIR / "cached_papers.json"

TEST_ARXIV_IDS = [
    "2005.11401",  # RAG paper
    "1706.03762",  # Attention is All You Need
    "2310.06825",  # Mistral 7B
]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires GPU/external services")
    config.addinivalue_line("markers", "gpu: requires GPU")


@pytest.fixture(scope="session")
def cached_papers():
    """Fetch papers from arXiv and cache locally. Downloads only on first run."""
    if CACHED_PAPERS_FILE.exists():
        with open(CACHED_PAPERS_FILE) as f:
            return json.load(f)

    from src.services import fetch_paper_html

    papers = []
    for arxiv_id in TEST_ARXIV_IDS:
        try:
            html_data = fetch_paper_html(arxiv_id)
            if html_data and html_data.success and html_data.full_text:
                papers.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": html_data.title,
                        "text": html_data.full_text[:10000],
                    }
                )
        except Exception:
            continue

    if not papers:
        pytest.skip("Could not fetch any papers from arXiv")

    FIXTURES_DIR.mkdir(exist_ok=True)
    with open(CACHED_PAPERS_FILE, "w") as f:
        json.dump(papers, f)

    return papers


@pytest.fixture(scope="session")
def chunked_papers(cached_papers):
    """Chunk cached papers into documents."""
    from src.services import chunk_document

    all_chunks = []
    for paper in cached_papers:
        pages = [{"text": paper["text"], "page": 1}]
        chunks = chunk_document(
            pages,
            source=f"{paper['arxiv_id']}.html",
            extra_metadata={
                "title": paper["title"],
                "arxiv_id": paper["arxiv_id"],
                "arxiv_url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                "authors": "Test Authors",
            },
        )
        all_chunks.extend(chunks)
    return all_chunks


@pytest.fixture(scope="session")
def test_vector_store(chunked_papers):
    """Create a test vector store with embedded papers. Requires Qdrant running."""
    from src.services.embeddings import get_embedder
    from src.services.vector_store import VectorStore

    embedder = get_embedder("qwen3-0.6b", use_flash_attn=False)
    vs = VectorStore(collection_prefix="test_cached", embedder=embedder)
    vs.clear()
    vs.add_documents(chunked_papers)

    yield vs

    vs.client.delete_collection(vs.collection_name)


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns random vectors (qwen3-0.6b dimensions)."""
    embedder = Mock()
    embedder.name = "mock-embedder"
    embedder.dimensions = 1024

    def encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [np.random.rand(1024).tolist() for _ in texts]

    def encode_query(query):
        return np.random.rand(1024).tolist()

    embedder.encode = encode
    embedder.encode_query = encode_query
    return embedder


@pytest.fixture
def mock_reranker():
    """Mock reranker that returns sorted by original order."""
    reranker = Mock()

    def rerank(query, documents, top_k=None):
        results = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = 1.0 - (i * 0.1)
            doc_copy["original_score"] = doc.get("score", 0.0)
            results.append(doc_copy)
        if top_k:
            results = results[:top_k]
        return results

    reranker.rerank = rerank
    return reranker


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = Mock()

    # Storage for points
    storage = {}

    def get_collections():
        result = Mock()
        result.collections = [Mock(name=name) for name in storage.keys()]
        return result

    def create_collection(collection_name, vectors_config):
        storage[collection_name] = []

    def delete_collection(collection_name):
        if collection_name in storage:
            del storage[collection_name]

    def upsert(collection_name, points):
        if collection_name not in storage:
            storage[collection_name] = []
        storage[collection_name].extend(points)

    def count(collection_name):
        result = Mock()
        result.count = len(storage.get(collection_name, []))
        return result

    def query_points(collection_name, query, limit):
        points = storage.get(collection_name, [])[:limit]
        result = Mock()
        result.points = [Mock(payload=p.payload.copy(), score=0.85) for p in points]
        return result

    client.get_collections = get_collections
    client.create_collection = create_collection
    client.delete_collection = delete_collection
    client.upsert = upsert
    client.count = count
    client.query_points = query_points

    return client


@pytest.fixture
def sample_documents():
    """Sample LangChain documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Retrieval augmented generation combines retrieval with LLMs.",
            metadata={"source": "paper1.pdf", "page": 1, "chunk": 0, "title": "RAG Paper"},
        ),
        Document(
            page_content="Transformers use self-attention mechanisms.",
            metadata={"source": "paper2.pdf", "page": 1, "chunk": 0, "title": "Attention Paper"},
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "content": "RAG combines retrieval with generation.",
            "metadata": {"title": "RAG Paper", "arxiv_url": "http://arxiv.org/abs/1234"},
            "score": 0.85,
        },
        {
            "content": "Transformers revolutionized NLP.",
            "metadata": {"title": "Transformer Paper", "arxiv_url": "http://arxiv.org/abs/5678"},
            "score": 0.75,
        },
    ]
