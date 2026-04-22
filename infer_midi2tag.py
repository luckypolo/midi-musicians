from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from midi2tag_model import (
    build_remi_tokenizer,
    collect_midi_paths,
    load_midi2tag_model,
    make_caption_from_tags,
    midi2tag_label_maps,
    tokenize_midi,
)


def predict_one(
    model,
    midi_path: Path,
    tokenizer,
    device: str,
    max_seq_len: int,
    track_policy: str,
) -> dict:
    token_ids = tokenize_midi(
        midi_path,
        tokenizer=tokenizer,
        track_policy=track_policy,
        max_seq_len=max_seq_len,
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    _, reverse_maps = midi2tag_label_maps(model.fields)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions: dict[str, str] = {}
    confidences: dict[str, float] = {}
    for field, field_logits in logits.items():
        probs = torch.softmax(field_logits[0], dim=-1)
        index = int(torch.argmax(probs).item())
        predictions[field] = reverse_maps[field][index]
        confidences[field] = float(probs[index].item())

    return {
        "midi_path": str(midi_path.resolve()),
        "predictions": predictions,
        "confidences": confidences,
        "caption": make_caption_from_tags(predictions),
        "token_count": len(token_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict simple text/control tags from MIDI files with a MIDI2Tags model.")
    parser.add_argument("path", help="MIDI file or directory of MIDI files.")
    parser.add_argument("--model-dir", default="artifacts/midi2tag")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--track-policy", choices=["single", "longest", "concat"], default="longest")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_midi2tag_model(args.model_dir, device)
    tokenizer = build_remi_tokenizer()
    midi_paths = collect_midi_paths(args.path)

    results = []
    failures = []
    for midi_path in midi_paths:
        try:
            results.append(
                predict_one(
                    model=model,
                    midi_path=midi_path,
                    tokenizer=tokenizer,
                    device=device,
                    max_seq_len=int(metadata.get("max_seq_len", 1024)),
                    track_policy=args.track_policy,
                )
            )
        except Exception as exc:
            failures.append({"midi_path": str(midi_path), "error": str(exc)})

    payload = {
        "device": device,
        "model_dir": str(Path(args.model_dir).resolve()),
        "track_policy": args.track_policy,
        "count": len(results),
        "results": results,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
