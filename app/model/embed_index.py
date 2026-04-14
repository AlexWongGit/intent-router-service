from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from app.model.schemas import IntentSample


@dataclass(slots=True)
class SearchHit:
    text: str
    intent: str
    score: float
    confidence_label: str


class EmbeddingIntentIndex:
    """
    负责：
    - 加载 embedding 模型
    - 加载 embeddings 索引（train_embeddings.npy）
    - 执行向量检索（cosine / dot）
    """

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self.model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )
        self.embeddings: np.ndarray | None = None
        self.samples: list[IntentSample] = []

    # =========================
    # 构建索引（训练阶段用）
    # =========================
    def build(self, samples: list[IntentSample]) -> None:
        self.samples = samples
        texts = [sample.text for sample in samples]

        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        self.embeddings = emb.astype(np.float32)

    # =========================
    # 保存索引
    # =========================
    def save(self, artifacts_dir: Path) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            raise RuntimeError("Index not built yet.")

        # 保存向量
        np.save(artifacts_dir / "train_embeddings.npy", self.embeddings)

        # 保存元数据
        metadata = [sample.model_dump() for sample in self.samples]
        (artifacts_dir / "train_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # =========================
    # 加载索引（服务启动用）
    # =========================
    def load(self, artifacts_dir: Path) -> None:
        emb_path = artifacts_dir / "train_embeddings.npy"
        meta_path = artifacts_dir / "train_metadata.json"

        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        emb = np.load(emb_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        self.embeddings = emb.astype(np.float32)
        self.samples = [IntentSample.model_validate(item) for item in metadata]

    # =========================
    # 向量检索
    # =========================
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if self.embeddings is None or not self.samples:
            raise RuntimeError("Index is empty.")

        # 编码 query
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        # 相似度（点积 = cosine，因为已 normalize）
        scores = self.embeddings @ q

        # 取 top-k
        top_indices = np.argsort(-scores)[:top_k]

        hits: list[SearchHit] = []
        for idx in top_indices:
            sample = self.samples[int(idx)]

            hits.append(
                SearchHit(
                    text=sample.text,
                    intent=sample.intent,
                    score=float(scores[idx]),
                    confidence_label=sample.confidence_label,
                )
            )

        return hits

    # =========================
    # 意图聚合（RRF-like）
    # =========================
    @staticmethod
    def aggregate_intent_scores(hits: list[SearchHit]) -> dict[str, float]:
        """
        把多个样本 hit 聚合成 intent 级别分数
        """
        bucket: dict[str, float] = defaultdict(float)

        for rank, hit in enumerate(hits, start=1):
            weight = 1.0 / rank
            bucket[hit.intent] += hit.score * weight

        return dict(
            sorted(bucket.items(), key=lambda x: x[1], reverse=True)
        )