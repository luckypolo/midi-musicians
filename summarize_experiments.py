from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_mean_accuracy(payload: dict[str, Any]) -> float | None:
    return payload.get("metrics", {}).get("mean_accuracy")


def extract_field_metrics(payload: dict[str, Any], fields: list[str]) -> dict[str, float | None]:
    metrics = payload.get("metrics", {})
    return {field: metrics.get(f"{field}_accuracy") for field in fields}


def extract_feature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("files", [])


def build_pairwise_deltas(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if len(rows) < 2:
        return {}
    numeric_fields = ["note_count", "notes_per_second", "avg_note_duration", "avg_velocity", "avg_pitch", "estimated_tempo"]
    baseline = rows[0]
    deltas: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        name = Path(row["path"]).stem
        deltas[name] = {}
        for field in numeric_fields:
            base_value = baseline.get(field)
            value = row.get(field)
            if isinstance(base_value, (int, float)) and isinstance(value, (int, float)):
                deltas[name][field] = round(float(value - base_value), 4)
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize control-adapter and MIDI-GPT experiment artifacts.")
    parser.add_argument("--before-eval", default="outputs/adapter_eval_before.json")
    parser.add_argument("--after-eval", default="outputs/adapter_eval_after.json")
    parser.add_argument("--augmented-eval", default="outputs/adapter_eval_augmented.json")
    parser.add_argument("--hybrid-payload", default="outputs/midigpt_bridge_payload_augmented_hybrid_v2.json")
    parser.add_argument("--hybrid-features", default="outputs/midigpt_bridge_run_augmented_hybrid_v2/feature_report.json")
    parser.add_argument("--output", default="outputs/experiment_summary.json")
    args = parser.parse_args()

    before_eval = load_json(args.before_eval)
    after_eval = load_json(args.after_eval)
    augmented_eval = load_json(args.augmented_eval)
    hybrid_payload = load_json(args.hybrid_payload)
    hybrid_features = load_json(args.hybrid_features)

    tracked_fields = ["instrument", "mood", "genre", "density_level", "polyphony_level", "note_duration_level", "register"]
    feature_rows = extract_feature_rows(hybrid_features)

    summary = {
        "adapter_comparison": {
            "baseline_mean_accuracy": extract_mean_accuracy(before_eval),
            "improved_mean_accuracy": extract_mean_accuracy(after_eval),
            "augmented_mean_accuracy": extract_mean_accuracy(augmented_eval),
            "baseline_fields": extract_field_metrics(before_eval, tracked_fields),
            "improved_fields": extract_field_metrics(after_eval, tracked_fields),
            "augmented_fields": extract_field_metrics(augmented_eval, tracked_fields),
        },
        "hybrid_predictions": [
            {
                "prompt": item["prompt"],
                "predictions": item["predictions"],
                "sources": item.get("sources", {}),
            }
            for item in hybrid_payload.get("items", [])
        ],
        "hybrid_feature_summary": hybrid_features.get("summary", {}),
        "hybrid_feature_rows": feature_rows,
        "pairwise_deltas_from_first_prompt": build_pairwise_deltas(feature_rows),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
