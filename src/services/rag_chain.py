from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections.abc import AsyncIterator
from .vector_store import VectorStore
from .reranker import Reranker

SYSTEM_PROMPT = """You are a research assistant that answers questions based on academic papers.
Use the provided context from research papers to answer the question.
If you cannot answer based on the context, say so clearly.

IMPORTANT: When citing information, use inline citation markers like [1], [2], etc. that correspond to the source numbers in the context. Every factual claim should have a citation.

Context:
{context}
"""

USER_PROMPT = """Question: {question}

Remember to use [1], [2], etc. citations inline when referencing information from the sources."""

FOLLOWUP_PROMPT = """Based on this Q&A about academic papers, suggest 3 brief follow-up questions the user might want to ask. Return only the questions, one per line, no numbering or bullets.

Question: {question}
Answer: {answer}

Follow-up questions:"""


class RAGChain:
    """RAG chain combining retrieval with LLM generation."""

    AVAILABLE_MODELS = {
        "qwen3:14b": "Qwen3 14B - Best balance for RAG",
        "gemma3:12b": "Gemma 3 12B - Best for chat",
        "mistral-small:24b": "Mistral Small 24B - High quality",
        "mistral-nemo": "Mistral Nemo 12B - Fast inference",
    }

    def __init__(
        self,
        model: str = "qwen3:14b",
        vector_store: VectorStore | None = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        temperature: float = 0.1,
        enable_reranking: bool = True,
        retrieval_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.model_name = model
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.vector_store = vector_store or VectorStore(host=qdrant_host, port=qdrant_port)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

        self.enable_reranking = enable_reranking
        self.retrieval_k = retrieval_k
        self.rerank_top_k = rerank_top_k

        self.reranker = Reranker() if enable_reranking else None

    def _retrieve_and_rerank(self, question: str, k: int = 5) -> list[dict]:
        """Two-stage retrieval: vector search + cross-encoder reranking."""
        fetch_k = self.retrieval_k if self.enable_reranking else k
        results = self.vector_store.search(question, k=fetch_k)

        if self.enable_reranking and self.reranker:
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        return results

    def _format_context(self, results: list[dict]) -> str:
        context_parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            title = meta.get("title", "Unknown")
            arxiv_url = meta.get("arxiv_url", "")
            authors = meta.get("authors", "Unknown")
            source_info = f"[{i}] {title}"
            if arxiv_url:
                source_info += f"\n    URL: {arxiv_url}"
            source_info += f"\n    Authors: {authors}"
            source_info += f"\n    Content: {r['content']}"
            context_parts.append(source_info)
        return "\n\n".join(context_parts)

    def query(self, question: str, k: int = 5) -> dict:
        """Query the RAG chain."""
        results = self._retrieve_and_rerank(question, k)
        context = self._format_context(results)
        answer = self.chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "sources": [
                {
                    "title": r["metadata"].get("title"),
                    "arxiv_url": r["metadata"].get("arxiv_url"),
                    "page": r["metadata"].get("page"),
                    "score": r.get("rerank_score", r.get("score", 0)),
                }
                for r in results
            ],
        }

    async def aquery(self, question: str, k: int = 5) -> dict:
        """Async query with two-stage retrieval."""
        results = self._retrieve_and_rerank(question, k)
        context = self._format_context(results)
        answer = await self.chain.ainvoke({"context": context, "question": question})

        sources = [
            {
                "id": i + 1,
                "title": r["metadata"].get("title"),
                "arxiv_url": r["metadata"].get("arxiv_url"),
                "authors": r["metadata"].get("authors"),
                "page": r["metadata"].get("page"),
                "content": r["content"][:500],
                "score": r.get("rerank_score", r.get("score", 0)),
                "original_score": r.get("original_score"),
            }
            for i, r in enumerate(results)
        ]

        return {"answer": answer, "sources": sources}

    async def aquery_stream(self, question: str, k: int = 5) -> tuple[list[dict], AsyncIterator[str]]:
        """Stream the answer while returning sources immediately."""
        results = self._retrieve_and_rerank(question, k)
        context = self._format_context(results)
        sources = [
            {
                "id": i + 1,
                "title": r["metadata"].get("title"),
                "arxiv_url": r["metadata"].get("arxiv_url"),
                "authors": r["metadata"].get("authors"),
                "page": r["metadata"].get("page"),
                "content": r["content"][:500],
                "score": r.get("rerank_score", r.get("score", 0)),
                "original_score": r.get("original_score"),
            }
            for i, r in enumerate(results)
        ]

        async def generate() -> AsyncIterator[str]:
            async for chunk in self.chain.astream({"context": context, "question": question}):
                yield chunk

        return sources, generate()

    async def generate_followups(self, question: str, answer: str) -> list[str]:
        """Generate follow-up question suggestions."""
        prompt = ChatPromptTemplate.from_template(FOLLOWUP_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        result = await chain.ainvoke({"question": question, "answer": answer})
        questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return questions[:3]
