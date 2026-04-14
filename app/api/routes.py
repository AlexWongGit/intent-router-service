from __future__ import annotations

from fastapi import APIRouter
from app.api.schemas import ReadyResponse
from app.core.logging import get_logger
from app.api.schemas import (
    HealthResponse,
    IntentPredictRequest,
    IntentPredictResponse,
    ScoreInfo,
    DebugInfo,
)
from app.service.intent_service import IntentService


router = APIRouter()
intent_service = IntentService()
logger = get_logger(__name__)

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    return ReadyResponse(
        status="ready" if intent_service.embedding_loaded and intent_service.classifier_loaded else "not_ready",
        embedding_loaded=intent_service.embedding_loaded,
        classifier_loaded=intent_service.classifier_loaded,
    )


@router.post("/intent/predict", response_model=IntentPredictResponse)
def predict_intent(request: IntentPredictRequest) -> IntentPredictResponse:
    logger.info(
        "Received intent predict request | query=%s | debug=%s",
        request.query,
        request.options.debug,
    )

    result = intent_service.predict(
        query=request.query,
        context=request.context,
        debug=request.options.debug,
    )

    debug_info = None
    if request.options.debug:
        debug_info = DebugInfo(
            enriched_query=result.enriched_query,
            risky_pair=result.risky_pair,
            extra={
                "embedding_hits": result.embedding_hits or [],
                "classifier_ranked_intents": result.classifier_ranked_intents or [],
            },
        )

    return IntentPredictResponse(
        top_intent=result.top_intent,
        route_target=result.route_target,
        need_clarification=result.need_clarification,
        clarification_question=result.clarification_question,
        decision_source=result.decision_source,
        scores=ScoreInfo(
            embedding_top=result.embedding_top,
            embedding_score=result.embedding_score,
            classifier_top=result.classifier_top,
            classifier_score=result.classifier_score,
        ),
        debug=debug_info,
    )