"""Embedding model abstraction for easy swapping and comparison."""
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    """Base class for embedding models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for collection naming."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""
        pass

    @abstractmethod
    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """Encode text(s) to vectors."""
        pass


class SentenceTransformerEmbedder(Embedder):
    """Wrapper for sentence-transformers models."""

    MODELS = {
        # Fast & small
        "minilm": "all-MiniLM-L6-v2",
        # Better quality
        "mpnet": "all-mpnet-base-v2",
        # BGE family (excellent for retrieval)
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        # E5 family
        "e5-small": "intfloat/e5-small-v2",
        "e5-base": "intfloat/e5-base-v2",
        "e5-large": "intfloat/e5-large-v2",
        # GTE (good balance)
        "gte-small": "thenlper/gte-small",
        "gte-base": "thenlper/gte-base",
        "gte-large": "thenlper/gte-large",
        # Scientific text
        "specter2": "allenai/specter2_base",
    }

    def __init__(self, model_key: str = "minilm"):
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")
        self._name = model_key
        self._model_id = self.MODELS[model_key]
        self._model = SentenceTransformer(self._model_id)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # E5 models require "query: " or "passage: " prefix
        if self._name.startswith("e5-"):
            texts = [f"passage: {t}" for t in texts]
        return self._model.encode(texts).tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a query (handles E5 prefix)."""
        if self._name.startswith("e5-"):
            query = f"query: {query}"
        return self._model.encode(query).tolist()


def get_embedder(model_key: str = "minilm") -> Embedder:
    """Factory function to get an embedder."""
    return SentenceTransformerEmbedder(model_key)


def list_available_models() -> dict[str, str]:
    """List all available embedding models."""
    return SentenceTransformerEmbedder.MODELS.copy()
