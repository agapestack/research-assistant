# Technology Stack

A detailed breakdown of every technology used and why it was chosen.

## Core Technologies

### Python 3.12+

The entire backend is Python, chosen for:

- Rich ML/AI ecosystem
- FastAPI async support
- Type hints for maintainability
- Extensive libraries for NLP

### uv Package Manager

```bash
# Instead of pip/poetry
uv sync        # Install dependencies
uv add package # Add new package
uv run script  # Run with venv
```

**Why uv?**

- 10-100x faster than pip
- Built-in virtual environment management
- Lock file for reproducibility
- Written in Rust

## Backend Framework

### FastAPI

```python
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = await rag.aquery(request.question, k=request.k)
    return QueryResponse(**result)
```

**Why FastAPI?**

| Feature | Benefit |
|---------|---------|
| Async native | Handle concurrent requests |
| Pydantic models | Automatic validation |
| OpenAPI docs | Auto-generated `/docs` |
| Type hints | IDE support, fewer bugs |

### Pydantic Settings

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    qdrant_host: str = "localhost"
    embedding_model: str = "qwen3-4b"
    llm_model: str = "qwen3:14b"
```

Configuration via environment variables with type safety.

## LLM Orchestration

### LangChain

Used for:

- Prompt templates
- Output parsing
- Chain composition

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | llm | StrOutputParser()
```

**Why LangChain?**

- Standard abstractions
- Easy LLM swapping
- Built-in streaming support

### Ollama

Local LLM inference:

```bash
ollama pull qwen3:14b
```

**Available Models**:

| Model | Size | Use Case |
|-------|------|----------|
| qwen3:14b | 14B | Best RAG balance |
| gemma3:12b | 12B | Good for chat |
| mistral-small:24b | 24B | High quality |
| mistral-nemo | 12B | Fast inference |

## Vector Database

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.create_collection(
    collection_name="papers_bge-base",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
```

**Why Qdrant?**

- Written in Rust (fast)
- Filtering support
- Easy Docker deployment
- Good Python client
- Free and open source

### Collection Strategy

Each embedding model gets its own collection:

```
papers_qwen3-4b    (2560 dims)
papers_qwen3-0.6b  (1024 dims)
papers_bge-large   (1024 dims)
```

This allows A/B testing different models.

## Embeddings

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
vectors = model.encode(["text1", "text2"])
```

Models selected based on [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

| Model | Dimensions | MTEB Retrieval |
|-------|-----------|----------------|
| Qwen3-Embedding-8B | 4096 | 70.58 |
| Qwen3-Embedding-4B | 4096 | ~68 |
| Qwen3-Embedding-0.6B | 4096 | ~65 |

### Reranker

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("jinaai/jina-reranker-v3", trust_remote_code=True)
results = model.rerank(query, documents)
```

Selected based on [MTEB Reranking leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Workflow Orchestration

### Prefect

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=30)
def search_arxiv(query: str, max_results: int):
    ...

@flow(name="collect-papers")
def collect_papers_flow(days_back: int = 30):
    results = search_arxiv(query, max_results)
    ...
```

**Why Prefect?**

- Python-native workflows
- Automatic retries
- Caching support
- Nice UI for monitoring

## Frontend

### SvelteKit 5

```svelte
<script>
  let answer = '';

  async function query() {
    const response = await fetch('/api/query', {...});
    for await (const chunk of streamResponse(response)) {
      answer += chunk;
    }
  }
</script>
```

**Why SvelteKit?**

- Minimal bundle size
- Great DX
- Built-in SSR
- Reactive by default

### Tailwind CSS

Utility-first styling with dark mode support.

## Data Storage

### SQLite

Tracking paper metadata and indexing status:

```sql
CREATE TABLE papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    is_indexed INTEGER DEFAULT 0,
    html_available INTEGER
);
```

**Why SQLite?**

- Zero configuration
- Single file
- Good enough for metadata
- Easy to backup

## Document Processing

### PyMuPDF (fitz)

PDF text extraction:

```python
import fitz
doc = fitz.open("paper.pdf")
text = doc[0].get_text()
```

### BeautifulSoup

HTML parsing for ar5iv:

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
sections = soup.find_all("section")
```

### LangChain Text Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)
```

## Development Tools

| Tool | Purpose |
|------|---------|
| pytest | Testing |
| ruff | Linting & formatting |

## Infrastructure

### Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
```

### GitHub Actions

CI/CD for:

- Linting (ruff)
- Building documentation
- Deploying to GitHub Pages
