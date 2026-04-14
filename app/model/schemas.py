from typing import Literal
from pydantic import BaseModel, Field


ConfidenceLabel = Literal["clear", "ambiguous", "hard"]


class IntentSample(BaseModel):
    text: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    confidence_label: ConfidenceLabel
    hard_negative: list[str] = Field(default_factory=list)