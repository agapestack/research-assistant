"""Tests for embeddings service."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSentenceTransformerEmbedder:
    @patch("src.services.embeddings.SentenceTransformer")
    def test_init_with_valid_model(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("minilm", use_flash_attn=False)

        assert embedder.name == "minilm"
        assert embedder.dimensions == 384

    def test_init_with_invalid_model(self):
        from src.services.embeddings import SentenceTransformerEmbedder

        with pytest.raises(ValueError, match="Unknown model"):
            SentenceTransformerEmbedder("nonexistent-model")

    @patch("src.services.embeddings.SentenceTransformer")
    def test_encode_single_text(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("minilm", use_flash_attn=False)
        result = embedder.encode("test text")

        assert len(result) == 1
        assert len(result[0]) == 384

    @patch("src.services.embeddings.SentenceTransformer")
    def test_encode_multiple_texts(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("minilm", use_flash_attn=False)
        result = embedder.encode(["text1", "text2"])

        assert len(result) == 2

    @patch("src.services.embeddings.SentenceTransformer")
    def test_encode_query(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("minilm", use_flash_attn=False)
        result = embedder.encode_query("test query")

        assert len(result) == 384

    @patch("src.services.embeddings.SentenceTransformer")
    def test_e5_model_adds_passage_prefix(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("e5-base", use_flash_attn=False)
        embedder.encode("test text")

        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0].startswith("passage: ")

    @patch("src.services.embeddings.SentenceTransformer")
    def test_e5_model_adds_query_prefix(self, mock_st_class):
        from src.services.embeddings import SentenceTransformerEmbedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder("e5-base", use_flash_attn=False)
        embedder.encode_query("test query")

        call_args = mock_model.encode.call_args[0][0]
        assert call_args.startswith("query: ")


class TestGetEmbedder:
    @patch("src.services.embeddings.SentenceTransformer")
    def test_get_embedder_returns_embedder(self, mock_st_class):
        from src.services.embeddings import Embedder, get_embedder

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = get_embedder("minilm", use_flash_attn=False)

        assert isinstance(embedder, Embedder)


class TestListAvailableModels:
    def test_list_available_models(self):
        from src.services.embeddings import list_available_models

        models = list_available_models()

        assert isinstance(models, dict)
        assert "minilm" in models
        assert "qwen3-4b" in models
        assert "bge-base" in models
