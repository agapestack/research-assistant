# API Reference

Complete documentation of the REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Query

#### `POST /query`

Query the RAG system with a natural language question.

**Request:**

```json
{
  "question": "What is retrieval augmented generation?",
  "k": 5,
  "model": "qwen3:14b"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | The question to ask |
| `k` | integer | 5 | Number of sources to retrieve |
| `model` | string | null | LLM model (null = default) |

**Response:**

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines...[1]...[2]",
  "sources": [
    {
      "id": 1,
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "arxiv_url": "https://arxiv.org/abs/2005.11401",
      "authors": "Lewis et al.",
      "page": 2,
      "content": "We explore a general-purpose fine-tuning recipe...",
      "score": 0.8234,
      "original_score": 0.7123
    }
  ],
  "model": "qwen3:14b"
}
```

---

#### `POST /query/stream`

Stream the response using Server-Sent Events.

**Request:** Same as `/query`

**Response:** SSE stream

```
data: {"type": "sources", "sources": [...], "model": "qwen3:14b"}

data: {"type": "chunk", "content": "Retrieval"}
data: {"type": "chunk", "content": "-Augmented"}
data: {"type": "chunk", "content": " Generation"}
...
data: {"type": "done"}
```

**JavaScript Example:**

```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "What is RAG?" })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'chunk') {
        console.log(data.content);
      }
    }
  }
}
```

---

### Search

#### `POST /search`

Semantic search without LLM generation.

**Request:**

```json
{
  "query": "transformer attention mechanism",
  "k": 10
}
```

**Response:**

```json
[
  {
    "content": "The Transformer architecture relies entirely on self-attention...",
    "title": "Attention Is All You Need",
    "arxiv_url": "https://arxiv.org/abs/1706.03762",
    "page": 3,
    "score": 0.8456
  }
]
```

---

### Index

#### `POST /index`

Fetch papers from arXiv and index them.

**Request:**

```json
{
  "topic": "retrieval augmented generation",
  "max_results": 10
}
```

**Response:**

```json
{
  "message": "Indexed papers on 'retrieval augmented generation'",
  "papers_found": 10,
  "papers_indexed": 8,
  "documents_indexed": 423
}
```

!!! note
    `papers_indexed` may be less than `papers_found` if some papers don't have ar5iv HTML available.

---

### Follow-ups

#### `POST /followups`

Generate follow-up question suggestions.

**Request:**

```json
{
  "question": "What is RAG?",
  "answer": "RAG combines retrieval with generation..."
}
```

**Response:**

```json
{
  "questions": [
    "How does RAG compare to fine-tuning?",
    "What are the limitations of RAG?",
    "Which vector databases work best with RAG?"
  ]
}
```

---

### Models

#### `GET /models`

List available LLM models.

**Response:**

```json
{
  "default": "qwen3:14b",
  "available": {
    "qwen3:14b": "Qwen3 14B - Best balance for RAG",
    "gemma3:12b": "Gemma 3 12B - Best for chat",
    "mistral-small:24b": "Mistral Small 24B - High quality",
    "mistral-nemo": "Mistral Nemo 12B - Fast inference"
  }
}
```

---

### Stats

#### `GET /stats`

Get vector store statistics.

**Response:**

```json
{
  "total_documents": 45230,
  "collection_name": "papers_bge-base"
}
```

---

### Health

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Unknown model. Available: ['qwen3:14b', 'gemma3:12b', ...]"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "Connection to Qdrant failed"
}
```

---

## Rate Limiting

No rate limiting is implemented. For production, consider:

- API key authentication
- Request rate limiting
- Response caching

---

## CORS

Allowed origins:

```python
allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"]
```

Modify in `src/main.py` for production.

---

## OpenAPI Documentation

Interactive docs available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
