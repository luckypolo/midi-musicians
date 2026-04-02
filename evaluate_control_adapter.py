from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_control_adapter import JsonDataset, collate_fn, evaluate
from control_adapter import load_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained control adapter on a held-out JSON split.")
    parser.add_argument("--model-dir", default="artifacts/control_adapter")
    parser.add_argument("--data", default="data/control_adapter/test.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_adapter(args.model_dir, device)
    dataset = JsonDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    metrics = evaluate(model, loader, device)
    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "data": str(Path(args.data).resolve()),
        "device": device,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
