from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FEATURES = [
    "note_count",
    "notes_per_second",
    "avg_note_duration",
    "avg_velocity",
    "avg_pitch",
    "estimated_tempo",
    "max_polyphony",
    "pitch_span",
]

CONTROL_FIELDS = [
    "density_level",
    "polyphony_level",
    "note_duration_level",
    "register",
    "mood",
    "genre",
    "instrument",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_items(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in payload["items"]}


def compare_items(v2: dict[str, Any], v3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item_id, old in v2.items():
        new = v3[item_id]
        changed_controls = {
            field: {"from": old["predictions"][field], "to": new["predictions"][field]}
            for field in CONTROL_FIELDS
            if old["predictions"][field] != new["predictions"][field]
        }
        feature_deltas = {}
        for field in NUMERIC_FEATURES:
            old_val = old["features"].get(field) if old.get("features") else None
            new_val = new["features"].get(field) if new.get("features") else None
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                feature_deltas[field] = round(float(new_val - old_val), 4)

        rows.append(
            {
                "id": item_id,
                "prompt": new["prompt"],
                "return_code_v2": old["return_code"],
                "return_code_v3": new["return_code"],
                "changed_controls": changed_controls,
                "feature_deltas_v3_minus_v2": feature_deltas,
            }
        )
    return rows


def aggregate_by_label(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for field in CONTROL_FIELDS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in payload["items"]:
            buckets[item["predictions"][field]].append(item["features"])
        grouped[field] = dict(buckets)

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for field, buckets in grouped.items():
        aggregated[field] = {}
        for label, features in buckets.items():
            feature_rows = [row for row in features if row]
            aggregated[field][label] = {
                metric: round(sum(float(row[metric]) for row in feature_rows) / max(len(feature_rows), 1), 4)
                for metric in ["note_count", "notes_per_second", "avg_note_duration", "avg_pitch", "estimated_tempo", "max_polyphony"]
            }
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark summary files and extract control/feature deltas.")
    parser.add_argument("--v2", default="outputs/benchmark_hybrid/benchmark_summary.json")
    parser.add_argument("--v3", default="outputs/benchmark_hybrid_v3/benchmark_summary.json")
    parser.add_argument("--output", default="outputs/benchmark_comparison_v2_v3.json")
    args = parser.parse_args()

    v2 = load_json(args.v2)
    v3 = load_json(args.v3)
    v2_items = index_items(v2)
    v3_items = index_items(v3)

    comparison = {
        "summary_v2": v2["summary"],
        "summary_v3": v3["summary"],
        "prompt_deltas": compare_items(v2_items, v3_items),
        "label_feature_means_v2": aggregate_by_label(v2),
        "label_feature_means_v3": aggregate_by_label(v3),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
