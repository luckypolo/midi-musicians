from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from control_adapter import CONTROL_FIELDS


PROGRAM_TO_FAMILY = {
    "piano": set(range(0, 8)),
    "organ": set(range(16, 24)),
    "guitar": set(range(24, 32)),
    "bass": set(range(32, 40)),
    "strings": set(range(40, 56)),
    "brass": set(range(56, 64)),
    "woodwind": set(range(72, 80)),
    "synth": set(range(80, 104)),
}

MOOD_BUCKETS = {
    "happy": {"happy", "uplifting", "joyful", "bright", "positive", "playful"},
    "sad": {"sad", "melancholic", "emotional", "romantic", "tender"},
    "dark": {"dark", "tense", "ominous", "mysterious", "grim"},
    "calm": {"calm", "peaceful", "relaxing", "meditative", "soft", "gentle"},
    "energetic": {"energetic", "driving", "lively", "exciting", "powerful"},
    "dramatic": {"epic", "cinematic", "dramatic", "film", "space"},
}

GENRE_BUCKETS = {
    "pop": {"pop"},
    "rock": {"rock"},
    "electronic": {"electronic", "dance", "house", "techno"},
    "classical": {"classical", "orchestral"},
    "jazz": {"jazz", "blues"},
    "cinematic": {"cinematic", "film", "soundtrack"},
    "ambient": {"ambient", "new age"},
    "folk": {"folk", "country"},
}


def clamp_label(value: str, field: str) -> str:
    return value if value in CONTROL_FIELDS[field] else CONTROL_FIELDS[field][-1]


def instrument_from_programs(programs: list[int] | None, instrument_summary: list[str] | None) -> str:
    counts = {field: 0 for field in CONTROL_FIELDS["instrument"]}
    for program in programs or []:
        if program >= 128:
            counts["drums"] += 2
            continue
        for family, domain in PROGRAM_TO_FAMILY.items():
            if program in domain:
                counts[family] += 1
                break

    text = " ".join((instrument_summary or [])).lower()
    if "drum" in text or "percussion" in text:
        counts["drums"] += 2
    if "cello" in text or "violin" in text or "string" in text:
        counts["strings"] += 1
    if "synth" in text:
        counts["synth"] += 1
    if "piano" in text or "keyboard" in text:
        counts["piano"] += 1

    best = max(counts.items(), key=lambda item: item[1])
    return "other" if best[1] == 0 else best[0]


def weighted_pick(values: list[str] | None, probs: list[float] | None, buckets: dict[str, set[str]], fallback: str) -> str:
    scores = {label: 0.0 for label in buckets}
    for value, prob in zip(values or [], probs or []):
        normalized = value.lower()
        for label, aliases in buckets.items():
            if normalized in aliases:
                scores[label] += float(prob)
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return fallback if best_score <= 0 else best_label


def density_from_metadata(row: dict[str, Any]) -> str:
    tempo = row.get("tempo") or 0
    inst_count = len(row.get("instrument_summary") or [])
    chord_count = len(row.get("all_chords") or [])
    duration = max(int(row.get("duration") or 1), 1)
    chord_rate = chord_count / duration
    score = 0
    if tempo >= 125:
        score += 1
    if inst_count >= 5:
        score += 1
    if chord_rate >= 0.35:
        score += 1
    if tempo <= 85:
        score -= 1
    if chord_rate <= 0.12:
        score -= 1
    if score <= -1:
        return "low"
    if score >= 2:
        return "high"
    return "medium"


def polyphony_from_metadata(row: dict[str, Any]) -> str:
    inst_count = len(row.get("instrument_summary") or [])
    chord_variety = len(set(row.get("all_chords") or []))
    if inst_count >= 5 or chord_variety >= 8:
        return "high"
    if inst_count <= 2 and chord_variety <= 3:
        return "low"
    return "medium"


def duration_from_metadata(row: dict[str, Any]) -> str:
    tempo = row.get("tempo") or 0
    tempo_word = str(row.get("tempo_word") or "").lower()
    if tempo <= 80 or tempo_word in {"largo", "adagio"}:
        return "long"
    if tempo >= 140 or tempo_word in {"presto", "vivace"}:
        return "short"
    return "medium"


def register_from_metadata(row: dict[str, Any]) -> str:
    programs = row.get("instrument_numbers_sorted") or []
    text = " ".join(row.get("instrument_summary") or []).lower()
    low_score = sum(1 for p in programs if 32 <= p < 44) + sum(
        token in text for token in ["bass", "contrabass", "cello", "tuba", "baritone"]
    )
    high_score = sum(1 for p in programs if p in {72, 73, 74, 75, 80, 81}) + sum(
        token in text for token in ["flute", "piccolo", "violin", "lead"]
    )
    if low_score > high_score:
        return "low"
    if high_score > low_score:
        return "high"
    return "medium"


def build_augmented_example(row: dict[str, Any]) -> dict[str, Any]:
    caption = row["caption"]
    return {
        "caption": caption,
        "instrument": clamp_label(
            instrument_from_programs(row.get("instrument_numbers_sorted"), row.get("instrument_summary")),
            "instrument",
        ),
        "mood": clamp_label(
            weighted_pick(row.get("mood"), row.get("mood_prob"), MOOD_BUCKETS, fallback="neutral"),
            "mood",
        ),
        "genre": clamp_label(
            weighted_pick(row.get("genre"), row.get("genre_prob"), GENRE_BUCKETS, fallback="other"),
            "genre",
        ),
        "density_level": density_from_metadata(row),
        "polyphony_level": polyphony_from_metadata(row),
        "note_duration_level": duration_from_metadata(row),
        "register": register_from_metadata(row),
        "source_location": row.get("location"),
        "tempo": row.get("tempo"),
        "key": row.get("key"),
        "time_signature": row.get("time_signature"),
        "duration_seconds": row.get("duration"),
        "test_set": bool(row.get("test_set", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a feature-augmented caption-to-controls dataset from MidiCaps.")
    parser.add_argument("--output-dir", default="data/control_adapter_augmented")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("amaai-lab/MidiCaps", split="train")
    rows = [build_augmented_example(dict(row)) for row in dataset]
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
