from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from midi2tag_model import load_midi2tag_model
from train_midi2tag import Midi2TagDataset, collate_fn, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MIDI2Tags model on a held-out JSON split.")
    parser.add_argument("--model-dir", default="artifacts/midi2tag")
    parser.add_argument("--data", default="data/midi2tag/test.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_midi2tag_model(args.model_dir, device)
    dataset = Midi2TagDataset(args.data, max_seq_len=int(metadata.get("max_seq_len", 1024)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_id=int(metadata.get("pad_id", 0))),
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
