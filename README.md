# AI Research Assistant

An intelligent research assistant that helps users explore, query, and synthesize information from academic papers and technical documentation.

## About This Project

This project was built to demonstrate practical experience with modern GenAI/LLM technologies, including:
- **RAG (Retrieval-Augmented Generation)** pipelines
- **Vector databases** for semantic search
- **LangChain/LangGraph** for LLM orchestration
- **Agentic workflows** for autonomous research tasks

As an AI Engineer with a background in computer vision and NLP, I built this to complement my production ML experience with hands-on GenAI application development.

**Author:** Jean Dié | [GitHub](https://github.com/agapestack) | [Email](mailto:jean.die@protonmail.com)

## Features

- **Document Ingestion:** Pipeline supporting PDF, markdown, and web content
- **Semantic Search:** Query research papers using vector similarity
- **Conversational Q&A:** Chat interface with source citations
- **Multi-document Analysis:** Summarization and comparison across papers
- **Agent Workflows:** Automated literature review and fact-checking

## Technical Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain / LangGraph |
| Vector Database | ChromaDB / Pinecone |
| Embeddings | Sentence-Transformers / OpenAI |
| LLM | GPT-4 / Claude / Mistral / Llama |
| Backend | FastAPI (async) |
| Frontend | Streamlit |
| Infrastructure | Docker, GitHub Actions |
| Observability | LangSmith / Weights & Biases |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   /ingest   │  │   /query    │  │      /agent         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   LangChain / LangGraph                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  RAG Chain  │  │   Agents    │  │       Tools         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└───────┬─────────────────┬───────────────────┬───────────────┘
        │                 │                   │
┌───────▼───────┐ ┌───────▼───────┐ ┌─────────▼─────────┐
│  Vector DB    │ │     LLM       │ │   External APIs   │
│  (ChromaDB)   │ │ (GPT/Claude)  │ │ (Arxiv, Web)      │
└───────────────┘ └───────────────┘ └───────────────────┘
```

## RAG Pipeline

1. **Document Processing**
   - PDF parsing with PyMuPDF
   - Text chunking (recursive, semantic)
   - Metadata extraction

2. **Embedding & Indexing**
   - Generate embeddings (sentence-transformers)
   - Store in vector database with metadata
   - Build hybrid index (dense + sparse)

3. **Retrieval**
   - Semantic search with similarity threshold
   - Hybrid search (BM25 + vector)
   - Reranking with cross-encoder

4. **Generation**
   - Context-aware prompt construction
   - LLM response with citations
   - Source verification

## Agent Capabilities

- **Literature Review Agent:** Searches arxiv, retrieves papers, summarizes findings
- **Fact-Checking Agent:** Verifies claims against indexed documents
- **Comparison Agent:** Analyzes differences between multiple papers

## Installation

```bash
git clone https://github.com/agapestack/ai-research-assistant.git
cd ai-research-assistant
pip install -r requirements.txt
```

## Usage

```bash
# Start the backend
uvicorn app.main:app --reload

# Start the UI
streamlit run ui/app.py
```

## Environment Variables

```bash
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

## Project Structure

```
ai-research-assistant/
├── src/
│   ├── main.py              # FastAPI application
│   ├── routers/
│   │   ├── ingest.py        # Document ingestion endpoints
│   │   ├── query.py         # RAG query endpoints
│   │   └── agent.py         # Agent endpoints
│   ├── services/
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── vectorstore.py   # Vector DB operations
│   │   ├── retriever.py     # Retrieval logic
│   │   └── chains.py        # LangChain chains
│   └── agents/
│       ├── researcher.py    # Research agent
│       └── tools.py         # Agent tools
├── ui/
│   └── app.py               # Streamlit interface
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Project setup
- [ ] Basic RAG pipeline (PDF → query)
- [ ] Vector database integration
- [ ] Reranking and citations
- [ ] Agent with arxiv tool
- [ ] Web search integration
- [ ] Streamlit UI
- [ ] Docker deployment
- [ ] CI/CD pipeline
