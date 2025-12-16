"""Tests for arXiv fetcher service."""

from datetime import datetime
from unittest.mock import Mock, patch


class TestSearchPapers:
    @patch("src.services.arxiv_fetcher.arxiv.Client")
    @patch("src.services.arxiv_fetcher.arxiv.Search")
    def test_search_returns_results(self, mock_search_class, mock_client_class):
        from src.services.arxiv_fetcher import search_papers

        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Test abstract"
        mock_result.authors = [Mock(name="Author One")]
        mock_result.published = datetime(2024, 1, 1)
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001"

        mock_client = Mock()
        mock_client.results.return_value = [mock_result]
        mock_client_class.return_value = mock_client

        results = search_papers("machine learning", max_results=5)

        assert len(results) == 1
        assert results[0].title == "Test Paper"
        mock_search_class.assert_called_once()

    @patch("src.services.arxiv_fetcher.arxiv.Client")
    @patch("src.services.arxiv_fetcher.arxiv.Search")
    def test_search_with_empty_query(self, mock_search_class, mock_client_class):
        from src.services.arxiv_fetcher import search_papers

        mock_client = Mock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client

        results = search_papers("", max_results=5)

        assert len(results) == 0

    @patch("src.services.arxiv_fetcher.arxiv.Client")
    @patch("src.services.arxiv_fetcher.arxiv.Search")
    def test_search_respects_max_results(self, mock_search_class, mock_client_class):
        from src.services.arxiv_fetcher import search_papers

        search_papers("test", max_results=10)

        mock_search_class.assert_called_once()
        call_kwargs = mock_search_class.call_args
        assert call_kwargs.kwargs["max_results"] == 10

    @patch("src.services.arxiv_fetcher.arxiv.Client")
    @patch("src.services.arxiv_fetcher.arxiv.Search")
    def test_search_uses_sort_criterion(self, mock_search_class, mock_client_class):
        import arxiv

        from src.services.arxiv_fetcher import search_papers

        mock_client = Mock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client

        search_papers("test", sort_by=arxiv.SortCriterion.SubmittedDate)

        call_kwargs = mock_search_class.call_args
        assert call_kwargs.kwargs["sort_by"] == arxiv.SortCriterion.SubmittedDate
