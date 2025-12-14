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
    embedding_model: str = "bge-base"
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
papers_minilm      (384 dims)
papers_bge-base    (768 dims)
papers_bge-large   (1024 dims)
```

This allows A/B testing different models.

## Embeddings

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
vectors = model.encode(["text1", "text2"])
```

**Model Options**:

| Model | Dimensions | Quality | Speed |
|-------|-----------|---------|-------|
| all-MiniLM-L6-v2 | 384 | Good | Fast |
| bge-base-en-v1.5 | 768 | Very Good | Medium |
| bge-large-en-v1.5 | 1024 | Excellent | Slow |

### Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([("query", "doc1"), ("query", "doc2")])
```

Cross-encoders are more accurate than bi-encoders but slower (can't pre-compute).

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
| ruff | Linting |
| mypy | Type checking |
| pre-commit | Git hooks |

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

- Running tests
- Building documentation
- Deploying to GitHub Pages
