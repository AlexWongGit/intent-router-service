from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.model.embed_index import EmbeddingIntentIndex


@dataclass(slots=True)
class PredictionResult:
    query: str
    top_intent: str
    top_score: float
    intent_scores: dict[str, float]
    ranked_intents: list[tuple[str, float]]
    hits: list[dict[str, Any]]
    need_clarification: bool
    rejected: bool


class IntentPredictor:
    def __init__(
        self,
        index: EmbeddingIntentIndex,
        top_k: int,
        reject_threshold: float,
        ambiguity_margin: float,
        reranker: Any | None = None,
        retrieval_top_k: int = 8,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.reject_threshold = reject_threshold
        self.ambiguity_margin = ambiguity_margin
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k

    @staticmethod
    def aggregate_intent_scores_from_hits(
        hits: list[dict[str, Any]],
    ) -> dict[str, float]:
        bucket: dict[str, float] = defaultdict(float)

        for rank, hit in enumerate(hits, start=1):
            weight = 1.0 / rank
            bucket[hit["intent"]] += hit["score"] * weight

        return dict(sorted(bucket.items(), key=lambda x: x[1], reverse=True))

    def predict(self, query: str) -> PredictionResult:
        retrieval_hits = self.index.search(query, top_k=self.retrieval_top_k)

        if self.reranker is not None:
            reranked_hits = self.reranker.rerank(query, retrieval_hits)
            normalized_hits = [
                {
                    "intent": hit.intent,
                    "text": hit.text,
                    "score": hit.score,
                    "retrieval_score": hit.retrieval_score,
                    "rerank_score": hit.rerank_score,
                    "confidence_label": hit.confidence_label,
                }
                for hit in reranked_hits[: self.top_k]
            ]
        else:
            normalized_hits = [
                {
                    "intent": hit.intent,
                    "text": hit.text,
                    "score": hit.score,
                    "retrieval_score": hit.score,
                    "rerank_score": None,
                    "confidence_label": hit.confidence_label,
                }
                for hit in retrieval_hits[: self.top_k]
            ]

        scores = self.aggregate_intent_scores_from_hits(normalized_hits)

        if not scores:
            return PredictionResult(
                query=query,
                top_intent="other",
                top_score=0.0,
                intent_scores={},
                ranked_intents=[],
                hits=[],
                need_clarification=False,
                rejected=True,
            )

        ranked = list(scores.items())
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        rejected = top_score < self.reject_threshold
        need_clarification = (
            not rejected
            and len(ranked) > 1
            and (top_score - second_score) < self.ambiguity_margin
        )

        if rejected:
            top_intent = "other"

        return PredictionResult(
            query=query,
            top_intent=top_intent,
            top_score=top_score,
            intent_scores=scores,
            ranked_intents=ranked,
            hits=normalized_hits,
            need_clarification=need_clarification,
            rejected=rejected,
        )