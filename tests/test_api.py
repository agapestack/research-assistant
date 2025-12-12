import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock


@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.count.return_value = 100
    store.collection_name = "papers"
    store.search.return_value = [
        {
            "content": "Sample content about RAG",
            "metadata": {
                "title": "Test Paper",
                "arxiv_url": "http://arxiv.org/abs/1234",
                "authors": "Test Author",
                "page": 1,
            },
            "score": 0.85,
        }
    ]
    return store


@pytest.fixture
def mock_rag_chain():
    chain = Mock()
    chain.model_name = "test-model"
    chain.AVAILABLE_MODELS = {"test-model": "Test Model"}
    chain.aquery = AsyncMock(
        return_value={
            "answer": "This is a test answer [1].",
            "sources": [
                {
                    "id": 1,
                    "title": "Test Paper",
                    "arxiv_url": "http://arxiv.org/abs/1234",
                    "authors": "Test Author",
                    "page": 1,
                    "content": "Sample content",
                    "score": 0.85,
                }
            ],
        }
    )
    chain.generate_followups = AsyncMock(
        return_value=["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
    )
    return chain


@pytest.fixture
def client(mock_vector_store, mock_rag_chain):
    with patch("src.main.get_vector_store", return_value=mock_vector_store):
        with patch("src.main.get_rag_chain", return_value=mock_rag_chain):
            from src.main import app
            yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelsEndpoint:
    def test_models_returns_available_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "default" in data
        assert "available" in data


class TestStatsEndpoint:
    def test_stats_returns_document_count(self, client, mock_vector_store):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 100
        assert data["collection_name"] == "papers"


class TestQueryEndpoint:
    def test_query_returns_answer_with_sources(self, client):
        response = client.post(
            "/query",
            json={"question": "What is RAG?", "k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model" in data
        assert len(data["sources"]) > 0

    def test_query_validates_empty_question(self, client):
        response = client.post(
            "/query",
            json={"question": "", "k": 5}
        )
        assert response.status_code == 422


class TestSearchEndpoint:
    def test_search_returns_results(self, client, mock_vector_store):
        response = client.post(
            "/search",
            json={"query": "transformer", "k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "content" in data[0]
        assert "title" in data[0]
        assert "score" in data[0]


class TestFollowupsEndpoint:
    def test_followups_returns_questions(self, client):
        response = client.post(
            "/followups",
            json={"question": "What is RAG?", "answer": "RAG is..."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert len(data["questions"]) == 3


class TestQueryStreamEndpoint:
    def test_stream_returns_sse(self, client, mock_rag_chain):
        mock_rag_chain.aquery_stream = AsyncMock(
            return_value=(
                [{"id": 1, "title": "Test", "arxiv_url": None, "authors": None, "page": 1, "content": "test", "score": 0.5}],
                async_generator_mock()
            )
        )
        with patch("src.main.get_rag_chain", return_value=mock_rag_chain):
            response = client.post(
                "/query/stream",
                json={"question": "What is RAG?", "k": 3}
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


async def async_generator_mock():
    yield "Test "
    yield "response"
