from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ClassifierPredictor:
    def __init__(
        self,
        model_dir: Path,
        artifacts_dir: Path,
        max_length: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.artifacts_dir = artifacts_dir
        self.max_length = max_length

        if not model_dir.exists():
            raise FileNotFoundError(f"Classifier model directory not found: {model_dir}")

        label2id_path = artifacts_dir / "label2id.json"
        id2label_path = artifacts_dir / "id2label.json"

        if not label2id_path.exists():
            raise FileNotFoundError(f"label2id.json not found: {label2id_path}")

        if not id2label_path.exists():
            raise FileNotFoundError(f"id2label.json not found: {id2label_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        with open(label2id_path, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)

        with open(id2label_path, "r", encoding="utf-8") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

        ranked = sorted(
            [(self.id2label[idx], prob) for idx, prob in enumerate(probs)],
            key=lambda x: x[1],
            reverse=True,
        )

        top_intent, top_score = ranked[0]

        return {
            "top_intent": top_intent,
            "top_score": float(top_score),
            "ranked_intents": ranked,
        }