# Architecture Overview

This document explains the system architecture, design decisions, and how components interact.

## System Diagram

```mermaid
flowchart TB
    subgraph Client
        UI[SvelteKit Frontend]
    end

    subgraph API["FastAPI Backend"]
        direction TB
        EP[Endpoints]
        RC[RAG Chain]
        VS[Vector Store]
        RR[Reranker]
    end

    subgraph Storage
        QD[(Qdrant)]
        SQL[(SQLite)]
    end

    subgraph External
        ARX[arXiv API]
        AR5[ar5iv HTML]
        OLL[Ollama LLM]
    end

    subgraph Workflows["Prefect Workflows"]
        COL[Collection Flow]
        IDX[Indexing Flow]
    end

    UI <-->|REST/SSE| EP
    EP --> RC
    RC --> VS
    RC --> RR
    RC --> OLL
    VS <--> QD

    COL --> ARX
    COL --> SQL
    IDX --> AR5
    IDX --> VS
    IDX --> SQL
```

## Component Responsibilities

### API Layer (`src/main.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | RAG query with sources |
| `/query/stream` | POST | Streaming response via SSE |
| `/search` | POST | Vector search only (no LLM) |
| `/index` | POST | Manual paper indexing |
| `/models` | GET | List available LLM models |
| `/stats` | GET | Vector store statistics |
| `/health` | GET | Health check |

### Services Layer (`src/services/`)

```
services/
├── embeddings.py      # Embedding model abstraction
├── vector_store.py    # Qdrant client wrapper
├── reranker.py        # Cross-encoder reranking
├── rag_chain.py       # RAG orchestration
├── document_loader.py # PDF/HTML → chunks
├── html_fetcher.py    # ar5iv HTML parsing
└── arxiv_fetcher.py   # arXiv API client
```

### Workflow Layer (`src/workflows/`)

Prefect flows for automated paper collection and indexing:

- **Collection Flow**: Search arXiv → deduplicate → save to SQLite
- **Indexing Flow**: Fetch HTML → chunk → embed → store in Qdrant
- **Orchestrator**: Combines collection + indexing

## Design Decisions

### Why ar5iv HTML over PDFs?

| Aspect | PDF | ar5iv HTML |
|--------|-----|------------|
| Text extraction | Noisy, layout issues | Clean, structured |
| Section boundaries | Hard to detect | Explicit `<section>` tags |
| Math rendering | Often corrupted | Proper LaTeX |
| Availability | All papers | ~80% of papers |

**Decision**: Use ar5iv as primary source, fall back gracefully when unavailable.

### Why Qdrant over alternatives?

| Database | Pros | Cons |
|----------|------|------|
| **Qdrant** | Fast, filtering, easy setup | Less ecosystem |
| ChromaDB | Simple, embedded | Performance at scale |
| Pinecone | Managed, scalable | Cost, vendor lock-in |
| Weaviate | Feature-rich | Complex setup |

**Decision**: Qdrant provides the best balance of performance, features, and simplicity for a self-hosted solution.

### Why Two-Stage Retrieval?

```mermaid
flowchart LR
    Q[Query] --> VS[Vector Search<br/>k=20, fast]
    VS --> RR[Reranker<br/>top_k=5, precise]
    RR --> LLM[LLM Generation]
```

1. **Stage 1 - Vector Search**: Fast approximate search over entire corpus (k=20)
2. **Stage 2 - Cross-Encoder**: Precise reranking of candidates (top_k=5)

This approach gets the best of both worlds:
- Speed of embedding-based search
- Precision of cross-encoder scoring

### Why Sentence Transformers over OpenAI?

| Aspect | Sentence Transformers | OpenAI Embeddings |
|--------|----------------------|-------------------|
| Cost | Free (local) | $0.0001/1K tokens |
| Latency | ~10ms | ~100-500ms |
| Privacy | Data stays local | Sent to API |
| Customization | Many models | Limited options |

**Decision**: Local embeddings for cost, speed, and privacy. OpenAI can be added as an option later.

## Data Flow

### Ingestion Pipeline

```mermaid
sequenceDiagram
    participant A as arXiv API
    participant S as SQLite
    participant H as ar5iv
    participant C as Chunker
    participant E as Embedder
    participant Q as Qdrant

    Note over A,Q: Collection Phase
    A->>S: Search results (metadata)
    S->>S: Deduplicate & store

    Note over A,Q: Indexing Phase
    S->>H: Get unindexed papers
    H->>C: HTML content
    C->>E: Text chunks
    E->>Q: Vectors + metadata
    Q->>S: Mark as indexed
```

### Query Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant VS as VectorStore
    participant RR as Reranker
    participant LLM as Ollama

    U->>API: Question
    API->>VS: Embed & search (k=20)
    VS->>API: Candidate chunks
    API->>RR: Rerank candidates
    RR->>API: Top 5 chunks
    API->>LLM: Context + question
    LLM-->>API: Streamed answer
    API-->>U: SSE chunks
```

## Scalability Considerations

### Current Limitations

- Single-node Qdrant (can be clustered)
- Synchronous embedding (can be batched)
- No caching layer (Redis can be added)

### Scaling Path

1. **More papers**: Qdrant handles millions of vectors
2. **More users**: Add Redis caching, load balancing
3. **Better quality**: Upgrade embedding model, add hybrid search
4. **Production**: Kubernetes deployment, monitoring
