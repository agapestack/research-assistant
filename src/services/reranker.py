"""Reranker for two-stage retrieval using jina-reranker-v3."""
from transformers import AutoModel


class Reranker:
    """Cross-encoder reranker using jina-reranker-v3."""

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3"):
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of dicts with 'content' key and optional metadata
            top_k: Number of top results to return (None = all, reordered)

        Returns:
            Documents sorted by reranker score, with 'rerank_score' added
        """
        if not documents:
            return []

        contents = [doc["content"] for doc in documents]
        results = self.model.rerank(query, contents, top_n=top_k)

        reranked = []
        for result in results:
            doc = documents[result["index"]].copy()
            doc["rerank_score"] = float(result["relevance_score"])
            doc["original_score"] = doc.get("score", 0.0)
            reranked.append(doc)

        return reranked
