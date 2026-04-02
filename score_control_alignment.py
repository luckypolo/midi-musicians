from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0
    low_idx = max(0, int(len(ordered) * 0.33) - 1)
    high_idx = min(len(ordered) - 1, int(len(ordered) * 0.66))
    return float(ordered[low_idx]), float(ordered[high_idx])


def bucket(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value >= high_cut:
        return "high"
    return "medium"


def bucket_duration(value: float, short_cut: float, long_cut: float) -> str:
    if value <= short_cut:
        return "short"
    if value >= long_cut:
        return "long"
    return "medium"


def score_item(item: dict[str, Any], cuts: dict[str, tuple[float, float]]) -> dict[str, Any]:
    features = item["features"]
    predicted = item["predictions"]

    observed = {
        "density_level": bucket(features["notes_per_second"], *cuts["notes_per_second"]),
        "polyphony_level": bucket(float(features["max_polyphony"]), *cuts["max_polyphony"]),
        "note_duration_level": bucket_duration(features["avg_note_duration"], *cuts["avg_note_duration"]),
        "register": bucket(features["avg_pitch"], *cuts["avg_pitch"]),
    }

    field_scores = {
        field: int(predicted[field] == observed[field])
        for field in observed
    }

    return {
        "id": item["id"],
        "predictions": {field: predicted[field] for field in observed},
        "observed": observed,
        "field_scores": field_scores,
        "alignment_score": sum(field_scores.values()) / len(field_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score how well control labels align with extracted MIDI features.")
    parser.add_argument("--benchmark", default="outputs/benchmark_hybrid_v3/benchmark_summary.json")
    parser.add_argument("--output", default="outputs/benchmark_alignment_v3.json")
    args = parser.parse_args()

    benchmark = load_json(args.benchmark)
    items = [item for item in benchmark["items"] if item.get("features")]

    cuts = {
        "notes_per_second": quantiles([float(item["features"]["notes_per_second"]) for item in items]),
        "max_polyphony": quantiles([float(item["features"]["max_polyphony"]) for item in items]),
        "avg_note_duration": quantiles([float(item["features"]["avg_note_duration"]) for item in items]),
        "avg_pitch": quantiles([float(item["features"]["avg_pitch"]) for item in items]),
    }

    scored = [score_item(item, cuts) for item in items]
    field_names = ["density_level", "polyphony_level", "note_duration_level", "register"]
    field_accuracy = {
        field: round(sum(row["field_scores"][field] for row in scored) / len(scored), 4)
        for field in field_names
    }

    payload = {
        "cuts": cuts,
        "field_accuracy": field_accuracy,
        "mean_alignment": round(sum(row["alignment_score"] for row in scored) / len(scored), 4),
        "items": scored,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
