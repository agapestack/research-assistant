"""Tests for reranker service."""

from unittest.mock import Mock, patch


class TestReranker:
    @patch("src.services.reranker.AutoModel")
    def test_init_loads_model(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        Reranker()

        mock_auto_model.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("src.services.reranker.AutoModel")
    def test_init_with_custom_model(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        Reranker(model_name="custom/model")

        call_args = mock_auto_model.from_pretrained.call_args
        assert call_args[0][0] == "custom/model"

    @patch("src.services.reranker.AutoModel")
    def test_rerank_empty_documents(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        reranker = Reranker()
        result = reranker.rerank("test query", [])

        assert result == []

    @patch("src.services.reranker.AutoModel")
    def test_rerank_returns_sorted_documents(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_model.rerank.return_value = [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.80},
        ]
        mock_auto_model.from_pretrained.return_value = mock_model

        reranker = Reranker()
        documents = [
            {"content": "Document A", "score": 0.7},
            {"content": "Document B", "score": 0.6},
        ]
        result = reranker.rerank("test query", documents)

        assert len(result) == 2
        assert result[0]["content"] == "Document B"
        assert result[0]["rerank_score"] == 0.95
        assert result[1]["content"] == "Document A"
        assert result[1]["rerank_score"] == 0.80

    @patch("src.services.reranker.AutoModel")
    def test_rerank_preserves_original_score(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_model.rerank.return_value = [
            {"index": 0, "relevance_score": 0.90},
        ]
        mock_auto_model.from_pretrained.return_value = mock_model

        reranker = Reranker()
        documents = [{"content": "Test", "score": 0.75}]
        result = reranker.rerank("query", documents)

        assert result[0]["original_score"] == 0.75
        assert result[0]["rerank_score"] == 0.90

    @patch("src.services.reranker.AutoModel")
    def test_rerank_with_top_k(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_model.rerank.return_value = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
        ]
        mock_auto_model.from_pretrained.return_value = mock_model

        reranker = Reranker()
        documents = [
            {"content": "Doc A"},
            {"content": "Doc B"},
            {"content": "Doc C"},
        ]
        reranker.rerank("query", documents, top_k=2)

        mock_model.rerank.assert_called_with("query", ["Doc A", "Doc B", "Doc C"], top_n=2)

    @patch("src.services.reranker.AutoModel")
    def test_rerank_does_not_mutate_input(self, mock_auto_model):
        from src.services.reranker import Reranker

        mock_model = Mock()
        mock_model.rerank.return_value = [
            {"index": 0, "relevance_score": 0.90},
        ]
        mock_auto_model.from_pretrained.return_value = mock_model

        reranker = Reranker()
        original_doc = {"content": "Test", "score": 0.75}
        documents = [original_doc]
        reranker.rerank("query", documents)

        assert "rerank_score" not in original_doc
