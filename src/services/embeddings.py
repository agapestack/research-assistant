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

    @abstractmethod
    def encode_query(self, query: str) -> list[float]:
        """Encode a query (may use different prompts/prefixes)."""
        pass


class SentenceTransformerEmbedder(Embedder):
    """Wrapper for sentence-transformers models."""

    MODELS = {
        # Qwen3 SOTA (requires transformers>=4.51.0, sentence-transformers>=2.7.0)
        "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
        "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
        "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
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

    # Models requiring special query handling
    _QWEN3_MODELS = {"qwen3-8b", "qwen3-4b", "qwen3-0.6b"}
    _E5_MODELS = {"e5-small", "e5-base", "e5-large"}

    def __init__(self, model_key: str = "qwen3-4b", use_flash_attn: bool = True):
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")
        self._name = model_key
        self._model_id = self.MODELS[model_key]

        model_kwargs = {"device_map": "auto"}
        tokenizer_kwargs = {}

        if model_key in self._QWEN3_MODELS:
            tokenizer_kwargs["padding_side"] = "left"
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        if model_kwargs.get("attn_implementation") or model_kwargs.get("device_map"):
            self._model = SentenceTransformer(
                self._model_id,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs else None,
            )
        else:
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
        if self._name in self._E5_MODELS:
            texts = [f"passage: {t}" for t in texts]
        return self._model.encode(texts).tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a query with appropriate prompts/prefixes."""
        if self._name in self._QWEN3_MODELS:
            return self._model.encode(query, prompt_name="query").tolist()
        if self._name in self._E5_MODELS:
            return self._model.encode(f"query: {query}").tolist()
        return self._model.encode(query).tolist()


def get_embedder(model_key: str = "qwen3-4b", use_flash_attn: bool = True) -> Embedder:
    """Factory function to get an embedder."""
    return SentenceTransformerEmbedder(model_key, use_flash_attn)


def list_available_models() -> dict[str, str]:
    """List all available embedding models."""
    return SentenceTransformerEmbedder.MODELS.copy()
