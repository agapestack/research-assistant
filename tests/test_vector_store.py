"""Tests for vector store service."""

from unittest.mock import Mock, patch

from langchain_core.documents import Document


class TestVectorStore:
    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_init_creates_collection(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test-embedder"
        mock_embedder.dimensions = 384
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()

        mock_client.create_collection.assert_called_once()
        assert vs.collection_name == "papers_test-embedder"

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_init_skips_existing_collection(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test-embedder"
        mock_embedder.dimensions = 384
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        existing_collection = Mock()
        existing_collection.name = "papers_test-embedder"
        mock_client.get_collections.return_value.collections = [existing_collection]
        mock_qdrant_class.return_value = mock_client

        VectorStore()

        mock_client.create_collection.assert_not_called()

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_add_documents_encodes_and_upserts(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test"
        mock_embedder.dimensions = 384
        mock_embedder.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()
        docs = [
            Document(page_content="Doc 1", metadata={"source": "a.pdf", "page": 1, "chunk": 0}),
            Document(page_content="Doc 2", metadata={"source": "b.pdf", "page": 1, "chunk": 0}),
        ]
        vs.add_documents(docs)

        mock_embedder.encode.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_add_documents_empty_list(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test"
        mock_embedder.dimensions = 384
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()
        vs.add_documents([])

        mock_client.upsert.assert_not_called()

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_search_encodes_query_and_returns_results(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test"
        mock_embedder.dimensions = 384
        mock_embedder.encode_query.return_value = [0.1] * 384
        mock_get_embedder.return_value = mock_embedder

        mock_point = Mock()
        mock_point.payload = {"content": "Test content", "title": "Test"}
        mock_point.score = 0.85

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_client.query_points.return_value.points = [mock_point]
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()
        results = vs.search("test query", k=5)

        mock_embedder.encode_query.assert_called_with("test query")
        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["score"] == 0.85

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_count_returns_document_count(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test"
        mock_embedder.dimensions = 384
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_client.count.return_value.count = 42
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()
        count = vs.count()

        assert count == 42

    @patch("src.services.vector_store.QdrantClient")
    @patch("src.services.vector_store.get_embedder")
    def test_clear_deletes_and_recreates(self, mock_get_embedder, mock_qdrant_class):
        from src.services.vector_store import VectorStore

        mock_embedder = Mock()
        mock_embedder.name = "test"
        mock_embedder.dimensions = 384
        mock_get_embedder.return_value = mock_embedder

        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant_class.return_value = mock_client

        vs = VectorStore()
        vs.clear()

        mock_client.delete_collection.assert_called_once()
        assert mock_client.create_collection.call_count == 2
