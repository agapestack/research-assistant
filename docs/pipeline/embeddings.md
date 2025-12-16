# Embedding Models

How text is converted to vectors for semantic search.

## What Are Embeddings?

Embeddings convert text into dense numerical vectors that capture semantic meaning:

```
"What is machine learning?" → [0.12, -0.34, 0.56, ..., 0.78]  (4096 dimensions)
```

Similar texts have similar vectors (high cosine similarity).

## Model Selection

Models are selected based on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) retrieval scores. We use Qwen3-Embedding as the default, which ranks #1 on MTEB multilingual retrieval (as of June 2025).

## Our Abstraction Layer

We built a configurable embedding system:

```python
from abc import ABC, abstractmethod

class Embedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for collection naming."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""

    @abstractmethod
    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """Encode text(s) to vectors."""

    @abstractmethod
    def encode_query(self, query: str) -> list[float]:
        """Encode a query (may use different prompts/prefixes)."""
```

### Switching Models

Change via environment variable:

```bash
# In .env
RAG_EMBEDDING_MODEL=qwen3-4b
```

Or pass directly:

```python
vs = VectorStore(embedding_model="qwen3-4b")
```

Each model creates its own collection: `papers_qwen3-0.6b`, `papers_qwen3-4b`, etc.

## Available Models

```python
MODELS = {
    # Qwen3 SOTA (requires transformers>=4.51.0, sentence-transformers>=2.7.0)
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",

    # Fast & small
    "minilm": "all-MiniLM-L6-v2",

    # BGE family
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",

    # E5 family
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",

    # GTE
    "gte-small": "thenlper/gte-small",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",

    # Scientific text
    "specter2": "allenai/specter2_base",
}
```

## Model Comparison

| Model | Dimensions | Parameters | MTEB Retrieval | VRAM |
|-------|-----------|------------|----------------|------|
| qwen3-8b | 4096 | 8B | **70.58** | ~16 GB |
| qwen3-4b | 2560 | 4B | ~68 | ~10 GB |
| qwen3-0.6b | 1024 | 0.6B | ~65 | ~2-3 GB |
| bge-large | 1024 | 335M | 54.29 | ~4 GB |
| bge-base | 768 | 110M | 53.25 | ~1 GB |
| minilm | 384 | 23M | ~42 | ~500 MB |

!!! tip "Recommendation"
    Use **qwen3-4b** as the default for excellent quality with reasonable VRAM (~10GB). Use **qwen3-0.6b** for lower resource environments.

## Qwen3 Model Features

- **Multilingual**: 100+ languages including code
- **Long context**: Up to 32K tokens
- **Flash Attention 2**: Enabled by default for memory efficiency
- **Instruction-aware**: Uses task-specific prompts for queries

### Special Query Handling

Qwen3 models use the `prompt_name="query"` parameter for queries:

```python
def encode_query(self, query: str) -> list[float]:
    if self._name in self._QWEN3_MODELS:
        return self._model.encode(query, prompt_name="query").tolist()
    return self._model.encode(query).tolist()
```

## E5 Model Special Handling

E5 models require prefixes:

```python
def encode(self, texts: str | list[str]) -> list[list[float]]:
    if self._name in self._E5_MODELS:
        texts = [f"passage: {t}" for t in texts]
    return self._model.encode(texts).tolist()

def encode_query(self, query: str) -> list[float]:
    if self._name in self._E5_MODELS:
        return self._model.encode(f"query: {query}").tolist()
    return self._model.encode(query).tolist()
```

## How to Choose

Models selected based on [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) retrieval scores.

| Use Case | Model |
|----------|-------|
| Prototyping | minilm |
| Production | qwen3-4b |
| Maximum quality | qwen3-8b |
| Low VRAM | qwen3-0.6b |

## Storage Considerations

| Model | Dims | Bytes/Vector | 100K Docs |
|-------|------|--------------|-----------|
| minilm | 384 | 1.5 KB | 150 MB |
| bge-base | 768 | 3 KB | 300 MB |
| qwen3-0.6b | 1024 | 4 KB | 400 MB |
| qwen3-4b | 2560 | 10 KB | 1 GB |

Qdrant handles millions of vectors efficiently with HNSW indexing.
