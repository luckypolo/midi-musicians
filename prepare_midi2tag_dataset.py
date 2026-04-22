from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from midi2tag_model import MIDI2TAG_FIELDS, build_remi_tokenizer, tokenize_midi
from prepare_augmented_control_dataset import build_augmented_example


DEFAULT_MIDI_ROOTS = [
    Path("data"),
    Path("data/lmd_full"),
    Path("data/midicaps_midis/lmd_full"),
    Path("data/midicaps_midis"),
    Path("data/Lakh MIDI Dataset/lmd_full"),
    Path("data/Lakh MIDI Dataset"),
]


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.metadata_json:
        data = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        return [dict(row) for row in data]
    dataset = load_dataset("amaai-lab/MidiCaps", split=args.split)
    return [build_augmented_example(dict(row)) for row in dataset]


def discover_midi_roots(midi_root: str | None) -> list[Path]:
    roots = [Path(midi_root)] if midi_root else []
    roots.extend(DEFAULT_MIDI_ROOTS)

    discovered: list[Path] = []
    for root in roots:
        if root.exists() and root not in discovered:
            discovered.append(root)
    return discovered


def resolve_midi_path(midi_roots: list[Path], row: dict[str, Any]) -> Path | None:
    candidates = [
        row.get("source_location"),
        row.get("location"),
        row.get("midi_path"),
        row.get("path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(str(candidate))
        paths = [candidate_path]
        for midi_root in midi_roots:
            stripped_candidate = (
                Path(*candidate_path.parts[1:])
                if candidate_path.parts and candidate_path.parts[0].lower() == midi_root.name.lower()
                else candidate_path
            )
            paths.extend(
                [
                    midi_root / candidate_path,
                    midi_root / stripped_candidate,
                    midi_root / candidate_path.name,
                ]
            )
        for path in paths:
            if path.exists():
                return path
    return None


def build_dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(args)
    random.Random(args.seed).shuffle(rows)

    tokenizer = build_remi_tokenizer()
    midi_roots = discover_midi_roots(args.midi_root)
    if not midi_roots:
        raise FileNotFoundError(
            "No MIDI root found. Pass --midi-root, or run download_lmd_full.py "
            "so data/lmd_full exists."
        )
    examples: list[dict[str, Any]] = []
    skipped = {
        "missing_midi": 0,
        "tokenization_failed": 0,
        "missing_label": 0,
    }

    for row in tqdm(rows, desc="Tokenizing MidiCaps rows"):
        if args.max_examples and len(examples) >= args.max_examples:
            break
        if any(field not in row for field in MIDI2TAG_FIELDS):
            skipped["missing_label"] += 1
            continue

        midi_path = resolve_midi_path(midi_roots, row)
        if midi_path is None:
            skipped["missing_midi"] += 1
            continue

        try:
            token_ids = tokenize_midi(
                midi_path,
                tokenizer=tokenizer,
                track_policy="single" if args.single_track_only else "longest",
                max_seq_len=args.max_seq_len,
            )
        except Exception:
            skipped["tokenization_failed"] += 1
            continue

        examples.append(
            {
                "midi_path": str(midi_path.resolve()),
                "source_location": row.get("source_location") or row.get("location"),
                "caption": row.get("caption", ""),
                "tokens": token_ids,
                "labels": {field: row[field] for field in MIDI2TAG_FIELDS},
                "test_set": bool(row.get("test_set", False)),
            }
        )

    summary = {
        "candidate_rows": len(rows),
        "examples": len(examples),
        "midi_roots": [str(root.resolve()) for root in midi_roots],
        "single_track_only": args.single_track_only,
        "max_seq_len": args.max_seq_len,
        "skipped": skipped,
    }
    return examples, summary


def split_examples(examples: list[dict[str, Any]], val_ratio: float, test_ratio: float) -> tuple[list, list, list]:
    explicit_test = [row for row in examples if row.get("test_set")]
    train_pool = [row for row in examples if not row.get("test_set")]

    if explicit_test:
        test_rows = explicit_test
    else:
        test_size = int(len(train_pool) * test_ratio)
        test_rows = train_pool[:test_size]
        train_pool = train_pool[test_size:]

    val_size = int(len(train_pool) * val_ratio)
    val_rows = train_pool[:val_size]
    train_rows = train_pool[val_size:]
    return train_rows, val_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a small MidiCaps MIDI2Tag dataset by tokenizing single-track MIDI files with MidiTok REMI."
    )
    parser.add_argument(
        "--midi-root",
        help="Root directory containing MidiCaps/Lakh MIDI files. Defaults to common data/lmd_full locations.",
    )
    parser.add_argument("--metadata-json", help="Optional prepared MidiCaps metadata JSON from data/control_adapter_augmented.")
    parser.add_argument("--split", default="train", help="Hugging Face split used when --metadata-json is omitted.")
    parser.add_argument("--output-dir", default="data/midi2tag")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-track-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    examples, summary = build_dataset_rows(args)
    train_rows, val_rows, test_rows = split_examples(examples, args.val_ratio, args.test_ratio)
    summary.update(
        {
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
            "fields": MIDI2TAG_FIELDS,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.json").write_text(json.dumps(train_rows, indent=2), encoding="utf-8")
    (output_dir / "val.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
    (output_dir / "test.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
