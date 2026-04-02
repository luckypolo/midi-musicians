from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from control_adapter import build_training_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a caption-to-controls dataset from MidiCaps.")
    parser.add_argument("--output-dir", default="data/control_adapter", help="Output directory for processed JSON files.")
    parser.add_argument("--max-train", type=int, default=5000, help="Maximum number of training examples.")
    parser.add_argument("--max-val", type=int, default=1000, help="Maximum number of validation examples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("amaai-lab/MidiCaps", split="train")
    rows = [build_training_example(dict(row)) for row in dataset]
    random.shuffle(rows)

    explicit_test = [row for row in rows if row["test_set"]]
    train_pool = [row for row in rows if not row["test_set"]]

    val_size = min(args.max_val, max(1, len(train_pool) // 10))
    train_size = min(args.max_train, max(1, len(train_pool) - val_size))

    train_rows = train_pool[:train_size]
    val_rows = train_pool[train_size : train_size + val_size]
    test_rows = explicit_test[: args.max_val]

    summary = {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
    }

    (output_dir / "train.json").write_text(json.dumps(train_rows, indent=2), encoding="utf-8")
    (output_dir / "val.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
    (output_dir / "test.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
