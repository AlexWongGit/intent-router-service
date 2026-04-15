from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api.schemas import IntentContext
from app.core.config import settings
from app.domain.policy import hybrid_decide

from app.model.embed_index import EmbeddingIntentIndex
from app.model.embedding_predictor import IntentPredictor
from app.model.classifier_predictor import ClassifierPredictor
from app.core.exceptions import ModelLoadException, PredictionException
from app.core.logging import get_logger


logger = get_logger(__name__)

INTENT_TO_ROUTE = {
    "ask_knowledge": "rag_answer",
    "ask_howto": "rag_howto",
    "query_status": "status_service",
    "summarize_content": "rag_summarize",
    "create_ticket": "agent_ticket",
    "update_ticket": "agent_update",
    "trigger_workflow": "agent_workflow",
    "smalltalk": "chat_fallback",
    "other": "fallback",
}


@dataclass(slots=True)
class IntentResult:
    top_intent: str
    route_target: str
    need_clarification: bool
    clarification_question: str | None
    decision_source: str
    embedding_top: str | None = None
    embedding_score: float | None = None
    classifier_top: str | None = None
    classifier_score: float | None = None
    enriched_query: str | None = None
    risky_pair: bool | None = None
    embedding_hits: list[dict] | None = None
    classifier_ranked_intents: list[tuple[str, float]] | None = None


class IntentService:
    def __init__(self) -> None:
        try:
            logger.info("Loading embedding index...")
            model_path = settings.embedding_model_dir
            self.embedding_index = EmbeddingIntentIndex(
                model_name=settings.embedding_model_dir,
                local_files_only=not model_path.startswith("BAAI"), # 从本地加载
            )
            self.embedding_index.load(Path(settings.artifacts_dir))
            self.embedding_loaded = True
            logger.info("Embedding index loaded successfully.")

            self.embedding_predictor = IntentPredictor(
                index=self.embedding_index,
                top_k=5,
                retrieval_top_k=8,
                reject_threshold=0.55,
                ambiguity_margin=settings.ambiguity_margin,
                reranker=None,
            )
            logger.info("Embedding predictor initialized.")

            self.classifier_predictor = ClassifierPredictor(
                model_dir=Path(settings.classifier_model_dir),
                artifacts_dir=Path(settings.artifacts_dir),
            )
            self.classifier_loaded = True
            logger.info("Classifier predictor initialized.")
        except Exception as exc:
            logger.exception("Failed to initialize IntentService.")
            self.embedding_loaded = False
            self.classifier_loaded = False
            raise ModelLoadException(str(exc)) from exc

    def enrich_query(self, query: str, context: IntentContext) -> str:
        if context.history:
            return f"上下文：{context.history[-1]}；当前问题：{query}"
        return query

    def predict(self, query: str, context: IntentContext, debug: bool = False) -> IntentResult:
        try:
            enriched_query = self.enrich_query(query, context)

            emb_result = self.embedding_predictor.predict(enriched_query)
            cls_result = self.classifier_predictor.predict(enriched_query)

            final = hybrid_decide(
                query=enriched_query,
                embedding_top=emb_result.top_intent,
                embedding_score=emb_result.top_score,
                embedding_ranked_intents=emb_result.ranked_intents,
                classifier_top=cls_result["top_intent"],
                classifier_score=cls_result["top_score"],
                ambiguity_margin=settings.ambiguity_margin,
                classifier_override_threshold=settings.classifier_override_threshold,
                risky_pair_override_threshold=settings.risky_pair_override_threshold,
            )

            route_target = INTENT_TO_ROUTE.get(final["top_intent"], "fallback")

            logger.info(
                "Intent predicted | query=%s | embedding_top=%s | classifier_top=%s | final=%s | clarification=%s | source=%s",
                query,
                emb_result.top_intent,
                cls_result["top_intent"],
                final["top_intent"],
                final["need_clarification"],
                final["source"],
            )

            return IntentResult(
                top_intent=final["top_intent"],
                route_target=route_target,
                need_clarification=final["need_clarification"],
                clarification_question=final["clarification_question"],
                decision_source=final["source"],
                embedding_top=emb_result.top_intent,
                embedding_score=emb_result.top_score,
                classifier_top=cls_result["top_intent"],
                classifier_score=cls_result["top_score"],
                enriched_query=enriched_query if debug else None,
                risky_pair=final["risky_pair"] if debug else None,
                embedding_hits=emb_result.hits[:3] if debug else None,
                classifier_ranked_intents=cls_result["ranked_intents"][:3] if debug else None,
            )
        except Exception as exc:
            logger.exception("Intent prediction failed | query=%s", query)
            raise PredictionException(str(exc)) from exc