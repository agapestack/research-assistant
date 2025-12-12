from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker for two-stage retrieval."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, max_length=512)

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

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0.0)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
