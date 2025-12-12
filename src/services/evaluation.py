import math
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


@dataclass
class EvaluationResult:
    """Complete evaluation of a RAG response."""
    relevance_score: float  # 0-1: How relevant is the answer to the question
    faithfulness_score: float  # 0-1: Is the answer grounded in the sources
    completeness_score: float  # 0-1: Does the answer fully address the question
    citation_accuracy: float  # 0-1: Are citations used correctly
    overall_score: float  # Weighted average
    reasoning: str  # LLM's explanation


@dataclass
class RetrievalMetrics:
    """Information retrieval quality metrics."""
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    precision_at_k: dict[int, float]  # Precision at k=1,3,5


JUDGE_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the following response based on the question and source documents provided.

Question: {question}

Sources:
{sources}

Answer: {answer}

Evaluate on these criteria (score 0.0-1.0 for each):

1. **Relevance**: Does the answer directly address the question asked?
2. **Faithfulness**: Is the answer grounded in the provided sources? No hallucinations?
3. **Completeness**: Does the answer fully address all aspects of the question?
4. **Citation Accuracy**: Are inline citations [1], [2], etc. used correctly to reference the sources?

Return your evaluation as JSON:
{{
    "relevance_score": <float 0-1>,
    "faithfulness_score": <float 0-1>,
    "completeness_score": <float 0-1>,
    "citation_accuracy": <float 0-1>,
    "reasoning": "<brief explanation of your scores>"
}}

Return ONLY valid JSON, no other text."""


class LLMJudge:
    """LLM-as-a-judge for evaluating RAG answer quality."""

    def __init__(self, model: str = "qwen3:14b", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        self.parser = JsonOutputParser()

    async def evaluate(
        self,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> EvaluationResult:
        """Evaluate a RAG response using LLM-as-a-judge."""
        sources_text = self._format_sources(sources)

        chain = self.prompt | self.llm | self.parser
        result = await chain.ainvoke({
            "question": question,
            "sources": sources_text,
            "answer": answer,
        })

        weights = {
            "relevance": 0.3,
            "faithfulness": 0.35,
            "completeness": 0.2,
            "citation": 0.15,
        }

        overall = (
            result["relevance_score"] * weights["relevance"]
            + result["faithfulness_score"] * weights["faithfulness"]
            + result["completeness_score"] * weights["completeness"]
            + result["citation_accuracy"] * weights["citation"]
        )

        return EvaluationResult(
            relevance_score=result["relevance_score"],
            faithfulness_score=result["faithfulness_score"],
            completeness_score=result["completeness_score"],
            citation_accuracy=result["citation_accuracy"],
            overall_score=round(overall, 3),
            reasoning=result["reasoning"],
        )

    def _format_sources(self, sources: list[dict]) -> str:
        parts = []
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            content = src.get("content", "")[:500]
            parts.append(f"[{i}] {title}\n{content}")
        return "\n\n".join(parts)


class RetrievalEvaluator:
    """Compute IR metrics for retrieval quality evaluation."""

    @staticmethod
    def compute_metrics(
        retrieved_docs: list[dict],
        relevance_scores: list[float] | None = None,
        k_values: list[int] = [1, 3, 5],
    ) -> RetrievalMetrics:
        """Compute MRR, NDCG, and Precision@K.

        Args:
            retrieved_docs: Documents with 'rerank_score' or 'score' fields
            relevance_scores: Optional ground-truth relevance (uses rerank_score if None)
            k_values: K values for precision calculation

        Returns:
            RetrievalMetrics with MRR, NDCG, P@K
        """
        if not retrieved_docs:
            return RetrievalMetrics(mrr=0.0, ndcg=0.0, precision_at_k={k: 0.0 for k in k_values})

        scores = relevance_scores or [
            doc.get("rerank_score", doc.get("score", 0.0)) for doc in retrieved_docs
        ]

        # Normalize scores to 0-1 range for metric computation
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0
        normalized = [(s - min_score) / score_range for s in scores]

        # Binary relevance threshold (top half of scores considered relevant)
        threshold = 0.5
        binary_relevance = [1 if s >= threshold else 0 for s in normalized]

        mrr = RetrievalEvaluator._compute_mrr(binary_relevance)
        ndcg = RetrievalEvaluator._compute_ndcg(normalized)
        precision_at_k = {k: RetrievalEvaluator._compute_precision_at_k(binary_relevance, k) for k in k_values}

        return RetrievalMetrics(
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            precision_at_k={k: round(v, 4) for k, v in precision_at_k.items()},
        )

    @staticmethod
    def _compute_mrr(binary_relevance: list[int]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant document."""
        for i, rel in enumerate(binary_relevance, 1):
            if rel == 1:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _compute_ndcg(relevance_scores: list[float], k: int | None = None) -> float:
        """Normalized Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0

        k = k or len(relevance_scores)
        scores = relevance_scores[:k]

        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _compute_precision_at_k(binary_relevance: list[int], k: int) -> float:
        """Precision@K: fraction of top-k docs that are relevant."""
        if k <= 0 or not binary_relevance:
            return 0.0
        top_k = binary_relevance[:k]
        return sum(top_k) / len(top_k)
