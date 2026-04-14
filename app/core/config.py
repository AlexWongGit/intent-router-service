from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "intent-router-service")
    service_port: int = int(os.getenv("SERVICE_PORT", "8080"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    embedding_model_dir: str = os.getenv(
        "EMBEDDING_MODEL_DIR",
        "models/embedding/bge-small-zh-v1.5",
    )
    classifier_model_dir: str = os.getenv(
        "CLASSIFIER_MODEL_DIR",
        "models/classifier/v3",
    )
    artifacts_dir: str = os.getenv(
        "ARTIFACTS_DIR",
        "models/artifacts",
    )

    ambiguity_margin: float = float(os.getenv("AMBIGUITY_MARGIN", "0.08"))
    classifier_override_threshold: float = float(
        os.getenv("CLASSIFIER_OVERRIDE_THRESHOLD", "0.80")
    )
    risky_pair_override_threshold: float = float(
        os.getenv("RISKY_PAIR_OVERRIDE_THRESHOLD", "0.55")
    )


settings = Settings()