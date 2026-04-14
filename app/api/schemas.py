from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadedFileInfo(BaseModel):
    name: str
    type: str


class IntentContext(BaseModel):
    history: list[str] = Field(default_factory=list)
    page_type: str | None = None
    object_type: str | None = None
    uploaded_files: list[UploadedFileInfo] = Field(default_factory=list)


class PredictOptions(BaseModel):
    debug: bool = False


class IntentPredictRequest(BaseModel):
    query: str
    context: IntentContext = Field(default_factory=IntentContext)
    options: PredictOptions = Field(default_factory=PredictOptions)


class ScoreInfo(BaseModel):
    embedding_top: str | None = None
    embedding_score: float | None = None
    classifier_top: str | None = None
    classifier_score: float | None = None


class DebugInfo(BaseModel):
    enriched_query: str | None = None
    risky_pair: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class IntentPredictResponse(BaseModel):
    top_intent: str
    route_target: str
    need_clarification: bool
    clarification_question: str | None = None
    decision_source: str
    scores: ScoreInfo
    debug: DebugInfo | None = None

class ReadyResponse(BaseModel):
    status: str
    embedding_loaded: bool
    classifier_loaded: bool

class HealthResponse(BaseModel):
    status: str