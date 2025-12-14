# Embedding Models

How text is converted to vectors for semantic search.

## What Are Embeddings?

Embeddings convert text into dense numerical vectors that capture semantic meaning:

```
"What is machine learning?" → [0.12, -0.34, 0.56, ..., 0.78]  (768 dimensions)
```

Similar texts have similar vectors (high cosine similarity).

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
```

### Switching Models

Change via environment variable:

```bash
# In .env
RAG_EMBEDDING_MODEL=bge-base
```

Or pass directly:

```python
vs = VectorStore(embedding_model="bge-base")
```

Each model creates its own collection: `papers_minilm`, `papers_bge-base`, etc.

## Available Models

```python
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
```

## Model Comparison

| Model | Dimensions | Size | MTEB Score | Speed |
|-------|-----------|------|------------|-------|
| minilm | 384 | 23MB | 0.42 | ⚡⚡⚡ |
| bge-small | 384 | 130MB | 0.51 | ⚡⚡⚡ |
| bge-base | 768 | 440MB | 0.53 | ⚡⚡ |
| bge-large | 1024 | 1.3GB | 0.54 | ⚡ |
| e5-base | 768 | 440MB | 0.52 | ⚡⚡ |
| specter2 | 768 | 440MB | - | ⚡⚡ |

!!! tip "Recommendation"
    Start with **bge-base** for the best quality/speed balance. Use **minilm** for prototyping.

## E5 Model Special Handling

E5 models require prefixes:

```python
def encode(self, texts: str | list[str]) -> list[list[float]]:
    if self._name.startswith("e5-"):
        texts = [f"passage: {t}" for t in texts]
    return self._model.encode(texts).tolist()

def encode_query(self, query: str) -> list[float]:
    if self._name.startswith("e5-"):
        query = f"query: {query}"
    return self._model.encode(query).tolist()
```

## Batch Encoding

For efficiency, we encode all chunks at once:

```python
def add_documents(self, documents: list[Document]) -> None:
    texts = [doc.page_content for doc in documents]
    vectors = self.embedder.encode(texts)  # Single batch call

    points = [
        PointStruct(id=self._generate_id(doc), vector=vec, payload={...})
        for doc, vec in zip(documents, vectors)
    ]
    self.client.upsert(collection_name=self.collection_name, points=points)
```

## How to Choose

### For Speed (Prototyping)

```bash
RAG_EMBEDDING_MODEL=minilm
```

- 384 dimensions
- ~10ms per query
- Good enough for testing

### For Quality (Production)

```bash
RAG_EMBEDDING_MODEL=bge-base
```

- 768 dimensions
- ~30ms per query
- Excellent retrieval quality

### For Academic Papers

```bash
RAG_EMBEDDING_MODEL=specter2
```

- Trained on scientific text
- Better for technical content
- May outperform general models

## Benchmarking

Run your own benchmarks:

```bash
# Compare specific models
uv run python scripts/benchmark_embeddings.py \
  --models minilm bge-base specter2 \
  --papers 10

# Test all models
uv run python scripts/benchmark_embeddings.py --all --papers 5
```

Output:

```
================================================================================
COMPARISON RESULTS
================================================================================
Model        Dims  Index(s)  Query(ms)    Top1      Top5
--------------------------------------------------------------------------------
bge-base      768     12.34      28.5   0.7234    0.6521
specter2      768     11.89      27.2   0.7156    0.6489
bge-small     384      8.45      15.3   0.6834    0.6123
minilm        384      5.67      12.1   0.6512    0.5834
--------------------------------------------------------------------------------

Best model: bge-base (top-5 score: 0.6521)
```

## Understanding Scores

### Cosine Similarity

```python
# Ranges from -1 to 1 (normalized to 0-1 in Qdrant)
similarity = dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec))
```

| Score | Interpretation |
|-------|----------------|
| 0.8+ | Very relevant |
| 0.6-0.8 | Relevant |
| 0.4-0.6 | Somewhat related |
| <0.4 | Probably irrelevant |

### Score Distribution

Good embeddings show clear separation:

```
Top 1: 0.82  ← Clear winner
Top 2: 0.71
Top 3: 0.68
Top 4: 0.54  ← Drop-off indicates relevance boundary
Top 5: 0.51
```

Bad embeddings are flat:

```
Top 1: 0.65
Top 2: 0.64
Top 3: 0.63  ← No discrimination
Top 4: 0.62
Top 5: 0.61
```

## Storage Considerations

| Model | Dims | Bytes/Vector | 100K Docs |
|-------|------|--------------|-----------|
| minilm | 384 | 1.5 KB | 150 MB |
| bge-base | 768 | 3 KB | 300 MB |
| bge-large | 1024 | 4 KB | 400 MB |

Qdrant handles millions of vectors efficiently with HNSW indexing.

## Future Improvements

### Hybrid Search

Combine semantic + keyword search:

```python
# BM25 for exact matches + embeddings for semantic
final_score = alpha * bm25_score + (1 - alpha) * semantic_score
```

### Fine-tuning

Train embeddings on your domain:

```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("bge-base-en-v1.5")
# Fine-tune on (query, relevant_doc, irrelevant_doc) triplets
```

### Quantization

Reduce storage with minimal quality loss:

```python
# Qdrant supports scalar quantization
vectors_config=VectorParams(
    size=768,
    distance=Distance.COSINE,
    quantization_config=ScalarQuantization(type=ScalarType.INT8)
)
```
