# Research Assistant

A RAG-powered research assistant for exploring and querying academic papers from arXiv.

## About

This project demonstrates practical experience with modern GenAI/LLM technologies:
- **RAG pipelines** with inline citations and streaming responses
- **Two-stage retrieval** with cross-encoder reranking (BAAI/bge-reranker-base)
- **LLM-as-a-judge** for automated answer quality evaluation
- **IR metrics** including MRR, NDCG, and Precision@K
- **Vector databases** for semantic search (Qdrant)
- **Workflow orchestration** with Prefect for automated paper collection
- **Local LLM inference** via Ollama
- **Full-stack development** with FastAPI backend and SvelteKit frontend

**Author:** Jean Dié | [GitHub](https://github.com/agapestack) | [Email](mailto:jean.die@protonmail.com)

## Features

- **Automated Paper Collection:** Prefect workflows fetch papers from arXiv on schedule
- **HTML-Based Ingestion:** Extract text from ar5iv (no PDF downloads required)
- **Two-Stage Retrieval:** Vector search → cross-encoder reranking for precision
- **RAG Q&A:** Ask questions and get answers with inline citations [1], [2]
- **Answer Quality Evaluation:** LLM-as-a-judge scores every response automatically
- **Retrieval Metrics:** MRR, NDCG, Precision@K computed per query
- **Streaming Responses:** Real-time answer generation via SSE
- **Multi-model Support:** Switch between Qwen3, Gemma, Mistral models
- **Modern UI:** SvelteKit frontend with dark mode, follow-up suggestions

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | SvelteKit 5, Tailwind CSS |
| Backend | FastAPI, SSE streaming |
| Orchestration | Prefect (workflow automation) |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | BAAI/bge-reranker-base (cross-encoder) |
| LLM | Ollama (Qwen3-14B, Gemma3-12B, Mistral) |
| Text Extraction | ar5iv HTML parsing (BeautifulSoup) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│              SvelteKit + Tailwind (localhost:5173)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      FastAPI Backend                             │
│  /query  /query/stream  /search  /index  /models  /stats        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       RAG Pipeline                               │
│  ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐ │
│  │ Vector     │──▶│ Reranker │──▶│ LLM Gen   │──▶│ LLM Judge │ │
│  │ Search     │   │ (BGE)    │   │ (Ollama)  │   │ (Eval)    │ │
│  └────────────┘   └──────────┘   └───────────┘   └───────────┘ │
│        │                │                              │        │
│        ▼                ▼                              ▼        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Metrics: MRR, NDCG, P@K  |  Scores: Relevance,         │   │
│  │                           |  Faithfulness, Completeness │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Prefect Workflows                             │
│  collect-papers → index-papers → daily-update → weekly-sync     │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/agapestack/research-assistant.git
cd research-assistant

# Install Python dependencies
uv sync

# Start infrastructure (Qdrant + Prefect)
docker compose up -d

# Pull an LLM model
ollama pull qwen3:14b

# Install frontend dependencies
cd frontend && npm install && cd ..
```

## Quick Start

### 1. Collect Papers
```bash
# Collect LLM/RAG papers from last 30 days
uv run python scripts/run_flow.py collect --days 30 --max-per-query 50
```

### 2. Index Papers
```bash
# Index collected papers (fetches HTML from ar5iv)
uv run python scripts/run_flow.py index --limit 20
```

### 3. Start the App
```bash
# Terminal 1: Backend
uv run uvicorn src.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit http://localhost:5173

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | RAG query with sources |
| `/query/stream` | POST | Streaming RAG response (SSE) |
| `/search` | POST | Semantic search only |
| `/index` | POST | Index papers by topic |
| `/followups` | POST | Generate follow-up questions |
| `/models` | GET | List available LLM models |
| `/stats` | GET | Vector store statistics |
| `/health` | GET | Health check |

## Prefect Workflows

```bash
# Run locally
uv run python scripts/run_flow.py collect   # Collect papers
uv run python scripts/run_flow.py index     # Index into vector store
uv run python scripts/run_flow.py full      # Full pipeline
uv run python scripts/run_flow.py daily     # Daily update preset
uv run python scripts/run_flow.py weekly    # Weekly sync preset

# With Prefect UI (http://localhost:4200)
prefect work-pool create default-agent-pool --type process
prefect deploy --all
prefect worker start --pool default-agent-pool
```

## Configuration

Environment variables (`.env`):

```bash
RAG_LLM_MODEL=qwen3:14b
RAG_QDRANT_HOST=localhost
RAG_QDRANT_PORT=6333
RAG_RETRIEVAL_K=5
RAG_LLM_TEMPERATURE=0.1
```

## Project Structure

```
research-assistant/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings (pydantic-settings)
│   ├── services/
│   │   ├── arxiv_fetcher.py    # arXiv API integration
│   │   ├── html_fetcher.py     # ar5iv HTML fetching
│   │   ├── document_loader.py  # Text extraction & chunking
│   │   ├── vector_store.py     # Qdrant operations
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── evaluation.py       # LLM-as-judge + IR metrics
│   │   └── rag_chain.py        # RAG chain with streaming
│   └── workflows/
│       ├── collection.py       # Paper collection flow
│       ├── indexing.py         # HTML indexing flow
│       └── orchestrator.py     # Combined pipelines
├── frontend/
│   ├── src/
│   │   ├── routes/+page.svelte # Main chat interface
│   │   └── lib/
│   │       ├── api.ts          # API client
│   │       └── components/     # UI components
│   └── package.json
├── scripts/
│   ├── run_flow.py             # Prefect flow CLI
│   ├── index_papers.py         # Manual indexing
│   └── query.py                # CLI query tool
├── tests/
│   ├── test_api.py             # API tests
│   └── test_document_loader.py # Document loader tests
├── data/
│   └── papers.db               # SQLite paper tracking
├── docker-compose.yml          # Qdrant + Prefect services
├── prefect.yaml                # Workflow deployments
└── pyproject.toml
```

## Supported Models

| Model | Size | VRAM | Notes |
|-------|------|------|-------|
| qwen3:14b | 14B | ~12GB | Default, best overall |
| gemma3:12b | 12B | ~10GB | Good for general Q&A |
| mistral-small:24b | 24B | ~16GB | Highest quality |
| mistral-nemo | 12B | ~8GB | Fast inference |

## Evaluation & Metrics

Every query returns evaluation scores and retrieval metrics:

### LLM-as-a-Judge Scores
| Metric | Description |
|--------|-------------|
| Relevance | Does the answer address the question? |
| Faithfulness | Is the answer grounded in sources (no hallucination)? |
| Completeness | Does it fully answer the question? |
| Citation Accuracy | Are [1], [2] citations used correctly? |

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank - position of first relevant result |
| NDCG | Normalized DCG - ranking quality considering position |
| P@K | Precision at K - fraction of top-K results that are relevant |

### Example Response
```json
{
  "answer": "...",
  "sources": [...],
  "evaluation": {
    "relevance_score": 0.9,
    "faithfulness_score": 0.85,
    "completeness_score": 0.8,
    "citation_accuracy": 0.75,
    "overall_score": 0.847,
    "reasoning": "The answer directly addresses..."
  },
  "retrieval_metrics": {
    "mrr": 1.0,
    "ndcg": 0.923,
    "precision_at_k": {"1": 1.0, "3": 0.667, "5": 0.6}
  }
}
```
