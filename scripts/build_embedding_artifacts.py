from __future__ import annotations

from pathlib import Path
import json

from app.model.embed_index import EmbeddingIntentIndex
from app.model.schemas import IntentSample


def load_jsonl(path: Path) -> list[IntentSample]:
    samples: list[IntentSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(IntentSample.model_validate(item))
    return samples


def main() -> None:
    train_file = Path("data/train_merged_v2.jsonl")
    artifacts_dir = Path("models/artifacts")
    model_name = "BAAI/bge-small-zh-v1.5"

    samples = load_jsonl(train_file)
    print(f"loaded train samples: {len(samples)}")

    index = EmbeddingIntentIndex(model_name=model_name, local_files_only=False)
    index.build(samples)
    index.save(artifacts_dir)

    print(f"embedding artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()