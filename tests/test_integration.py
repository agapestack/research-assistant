"""Integration tests requiring GPU and external services.

Run with: uv run pytest tests/test_integration.py -v -m integration
Requires: GPU, Qdrant running on localhost:6333

First run will download papers from arXiv and cache them locally.
"""

import pytest


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Test actual embedding models (requires GPU)."""

    @pytest.mark.gpu
    def test_qwen3_produces_vectors(self):
        """Test Qwen3-0.6B model produces correct dimension vectors."""
        from src.services.embeddings import get_embedder

        embedder = get_embedder("qwen3-0.6b", use_flash_attn=False)
        vectors = embedder.encode("Test sentence for embedding.")

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
        assert all(isinstance(v, float) for v in vectors[0])

    @pytest.mark.gpu
    def test_qwen3_query_encoding(self):
        """Test Qwen3 query encoding uses correct prompt."""
        from src.services.embeddings import get_embedder

        embedder = get_embedder("qwen3-0.6b", use_flash_attn=False)
        query_vec = embedder.encode_query("What is machine learning?")

        assert len(query_vec) == 1024
        assert all(isinstance(v, float) for v in query_vec)

    @pytest.mark.gpu
    def test_embed_cached_papers(self, cached_papers):
        """Test embedding actual paper content."""
        from src.services.embeddings import get_embedder

        embedder = get_embedder("qwen3-0.6b", use_flash_attn=False)
        paper_text = cached_papers[0]["text"][:500]
        vectors = embedder.encode(paper_text)

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024


@pytest.mark.integration
class TestRerankerIntegration:
    """Test actual reranker model (requires GPU)."""

    @pytest.mark.gpu
    def test_reranker_scores_documents(self):
        """Test reranker produces relevance scores."""
        from src.services.reranker import Reranker

        reranker = Reranker()
        documents = [
            {"content": "Python is a programming language."},
            {"content": "The weather is sunny today."},
            {"content": "Python was created by Guido van Rossum."},
        ]
        results = reranker.rerank("What is Python?", documents)

        assert len(results) == 3
        assert all("rerank_score" in r for r in results)
        python_docs = [r for r in results if "Python" in r["content"]]
        other_docs = [r for r in results if "Python" not in r["content"]]
        assert python_docs[0]["rerank_score"] > other_docs[0]["rerank_score"]

    @pytest.mark.gpu
    def test_reranker_with_paper_content(self, cached_papers):
        """Test reranker with actual paper content."""
        from src.services.reranker import Reranker

        reranker = Reranker()
        documents = [{"content": p["text"][:500]} for p in cached_papers]
        results = reranker.rerank("What is retrieval augmented generation?", documents)

        assert len(results) == len(cached_papers)
        assert results[0]["rerank_score"] > results[-1]["rerank_score"]


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Test with real Qdrant and cached papers."""

    def test_vector_store_count(self, test_vector_store):
        """Test vector store has documents from cached papers."""
        count = test_vector_store.count()
        assert count > 0

    def test_search_returns_relevant_results(self, test_vector_store):
        """Test search returns relevant chunks from papers."""
        results = test_vector_store.search("attention mechanism transformer", k=5)

        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_rag_query(self, test_vector_store):
        """Test searching for RAG-related content."""
        results = test_vector_store.search("retrieval augmented generation", k=3)

        assert len(results) > 0
        found_rag = any("retrieval" in r["content"].lower() for r in results)
        assert found_rag


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test full RAG pipeline with cached papers (requires GPU + Qdrant + Ollama)."""

    @pytest.mark.gpu
    def test_rag_query_with_cached_papers(self, test_vector_store):
        """Test full RAG query using cached paper data."""
        from src.services.rag_chain import RAGChain

        chain = RAGChain(vector_store=test_vector_store, enable_reranking=False)
        result = chain.query("What is the attention mechanism?")

        assert "answer" in result
        assert "sources" in result
        assert len(result["answer"]) > 0

    @pytest.mark.gpu
    def test_rag_query_with_reranking(self, test_vector_store):
        """Test RAG query with reranking enabled."""
        from src.services.rag_chain import RAGChain

        chain = RAGChain(vector_store=test_vector_store, enable_reranking=True)
        result = chain.query("How does RAG improve language models?")

        assert "answer" in result
        assert "sources" in result
