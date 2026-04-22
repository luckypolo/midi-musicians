from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from midi2tag_model import MIDI2TAG_FIELDS
from prompt_to_controls import parse_prompt


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    raw = load_json(path)
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if isinstance(raw, dict) and "caption" in raw:
        raw = [raw]

    prompts = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append({"id": f"prompt_{index:02d}", "caption": item})
        else:
            prompts.append(
                {
                    "id": str(item.get("id", f"prompt_{index:02d}")),
                    "caption": str(item["caption"]),
                }
            )
    return prompts


def intended_tags(caption: str) -> dict[str, str]:
    parsed = parse_prompt(caption)
    return {
        "mood": parsed.mood,
        "density_level": parsed.density_level,
        "polyphony_level": parsed.polyphony_level,
        "note_duration_level": parsed.note_duration_level,
        "register": parsed.register,
    }


def prompt_id_from_prediction(result: dict[str, Any], prompts: list[dict[str, str]]) -> str | None:
    stem = Path(result["midi_path"]).stem
    match = re.match(r"bridge_run_(\d+)$", stem)
    if match:
        index = int(match.group(1))
        return prompts[index]["id"] if index < len(prompts) else None

    prompt_ids = {prompt["id"] for prompt in prompts}
    return stem if stem in prompt_ids else None


def score_result(result: dict[str, Any], prompt: dict[str, str]) -> dict[str, Any]:
    intended = intended_tags(prompt["caption"])
    predicted = {field: result["predictions"][field] for field in MIDI2TAG_FIELDS}
    field_scores = {field: int(predicted[field] == intended[field]) for field in MIDI2TAG_FIELDS}
    return {
        "id": prompt["id"],
        "prompt": prompt["caption"],
        "midi_path": result["midi_path"],
        "intended": intended,
        "predicted": predicted,
        "confidences": {field: result.get("confidences", {}).get(field) for field in MIDI2TAG_FIELDS},
        "caption": result.get("caption"),
        "field_scores": field_scores,
        "alignment_score": round(sum(field_scores.values()) / len(MIDI2TAG_FIELDS), 4),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_alignment": 0.0,
            "field_accuracy": {field: 0.0 for field in MIDI2TAG_FIELDS},
        }
    return {
        "count": len(rows),
        "mean_alignment": round(sum(float(row["alignment_score"]) for row in rows) / len(rows), 4),
        "field_accuracy": {
            field: round(sum(int(row["field_scores"][field]) for row in rows) / len(rows), 4)
            for field in MIDI2TAG_FIELDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated MIDI2Tags predictions against the intended tags implied by the source prompts."
    )
    parser.add_argument("--predictions", required=True, help="JSON from infer_midi2tag.py.")
    parser.add_argument("--prompts", default="prompts_next_phase.json")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    prompts = load_prompts(args.prompts)
    prompts_by_id = {prompt["id"]: prompt for prompt in prompts}

    rows = []
    unmatched = []
    for result in predictions.get("results", []):
        prompt_id = prompt_id_from_prediction(result, prompts)
        if prompt_id is None or prompt_id not in prompts_by_id:
            unmatched.append(result["midi_path"])
            continue
        rows.append(score_result(result, prompts_by_id[prompt_id]))

    payload = {
        "predictions": str(Path(args.predictions).resolve()),
        "prompts": str(Path(args.prompts).resolve()),
        "summary": summarize(rows),
        "items": rows,
        "unmatched": unmatched,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
