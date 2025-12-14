import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import settings
from .services import RAGChain, VectorStore, load_paper_from_html
from .workflows.collection import search_arxiv


_vector_store: VectorStore | None = None
_rag_chain: RAGChain | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            embedding_model=settings.embedding_model,
        )
    return _vector_store


def get_rag_chain(model: str | None = None) -> RAGChain:
    global _rag_chain
    model = model or settings.llm_model
    if _rag_chain is None or _rag_chain.model_name != model:
        _rag_chain = RAGChain(
            model=model,
            vector_store=get_vector_store(),
            temperature=settings.llm_temperature,
        )
    return _rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Research Assistant",
    description="RAG-powered research assistant for academic papers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=settings.retrieval_k, ge=1, le=20)
    model: str | None = Field(default=None)


class Source(BaseModel):
    id: int
    title: str | None
    arxiv_url: str | None
    authors: str | None
    page: int | None
    content: str | None
    score: float
    original_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


class FollowupRequest(BaseModel):
    question: str
    answer: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    content: str
    title: str | None
    arxiv_url: str | None
    page: int | None
    score: float


class IndexRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)


class IndexResponse(BaseModel):
    message: str
    papers_found: int
    papers_indexed: int
    documents_indexed: int


class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system with a natural language question."""
    if request.model and request.model not in RAGChain.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model. Available: {list(RAGChain.AVAILABLE_MODELS.keys())}",
        )
    rag = get_rag_chain(request.model)
    result = await rag.aquery(request.question, k=request.k)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=rag.model_name,
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream the RAG response with Server-Sent Events."""
    if request.model and request.model not in RAGChain.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model. Available: {list(RAGChain.AVAILABLE_MODELS.keys())}",
        )
    rag = get_rag_chain(request.model)
    sources, stream = await rag.aquery_stream(request.question, k=request.k)

    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'model': rag.model_name})}\n\n"
        async for chunk in stream:
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/followups")
async def generate_followups(request: FollowupRequest):
    """Generate follow-up question suggestions."""
    rag = get_rag_chain()
    questions = await rag.generate_followups(request.question, request.answer)
    return {"questions": questions}


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    """Semantic search over indexed papers (no LLM generation)."""
    results = get_vector_store().search(request.query, k=request.k)
    return [
        SearchResult(
            content=r["content"],
            title=r["metadata"].get("title"),
            arxiv_url=r["metadata"].get("arxiv_url"),
            page=r["metadata"].get("page"),
            score=r["score"],
        )
        for r in results
    ]


@app.post("/index", response_model=IndexResponse)
async def index_papers(request: IndexRequest):
    """Fetch papers from arXiv and index them using HTML source."""
    query = f'ti:"{request.topic}" OR abs:"{request.topic}"'
    papers = search_arxiv.fn(query, request.max_results, days_back=365)

    documents_indexed = 0
    papers_indexed = 0

    for paper in papers:
        docs = load_paper_from_html(
            arxiv_id=paper["arxiv_id"],
            metadata={
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "published": paper["published"],
                "categories": ", ".join(paper["categories"]),
                "arxiv_url": paper["arxiv_url"],
            }
        )
        if docs:
            get_vector_store().add_documents(docs)
            documents_indexed += len(docs)
            papers_indexed += 1

    return IndexResponse(
        message=f"Indexed papers on '{request.topic}'",
        papers_found=len(papers),
        papers_indexed=papers_indexed,
        documents_indexed=documents_indexed,
    )


@app.get("/models")
async def list_models():
    """List available LLM models."""
    return {
        "default": settings.llm_model,
        "available": RAGChain.AVAILABLE_MODELS,
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics."""
    vs = get_vector_store()
    return StatsResponse(
        total_documents=vs.count(),
        collection_name=vs.collection_name,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
