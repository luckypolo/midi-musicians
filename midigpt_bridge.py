from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from control_adapter import load_adapter, predict_controls
from prompt_to_controls import parse_prompt


INSTRUMENT_TO_MIDIGPT = {
    "piano": "acoustic_grand_piano",
    "guitar": "electric_guitar_clean",
    "bass": "electric_bass_finger",
    "strings": "string_ensemble_1",
    "brass": "trumpet",
    "woodwind": "flute",
    "organ": "drawbar_organ",
    "synth": "lead_1_square",
    "drums": "standard_drum_kit",
    "other": "acoustic_grand_piano",
}

DENSITY_TO_SCORE = {"low": 3, "medium": 6, "high": 9}
POLYPHONY_TO_LIMIT = {"low": 2, "medium": 6, "high": 10}

HEURISTIC_INSTRUMENT_MAP = {
    "acoustic guitar": "guitar",
    "electric guitar": "guitar",
    "violin": "strings",
    "viola": "strings",
    "cello": "strings",
    "contrabass": "bass",
    "percussion": "drums",
}


def infer_track_type(predictions: dict[str, str]) -> str:
    if predictions["instrument"] == "drums":
        return "DRUM_TRACK"
    return "STANDARD_TRACK"


def temperature_from_predictions(predictions: dict[str, str]) -> float:
    base = 0.5
    mood = predictions["mood"]
    genre = predictions["genre"]
    density = predictions["density_level"]

    if mood in {"sad", "calm"}:
        base -= 0.08
    if mood in {"energetic", "dark", "dramatic"}:
        base += 0.08
    if genre in {"classical", "ambient"}:
        base -= 0.05
    if genre in {"electronic", "pop", "rock", "cinematic"}:
        base += 0.04
    if density == "low":
        base -= 0.04
    if density == "high":
        base += 0.05

    return max(0.55, min(0.9, round(base, 2)))


def normalize_heuristic_instrument(value: str) -> str:
    normalized = value.strip().lower()
    return HEURISTIC_INSTRUMENT_MAP.get(normalized, normalized if normalized in INSTRUMENT_TO_MIDIGPT else "other")


def resolve_predictions(prompt: str, learned_predictions: dict[str, str], confidences: dict[str, float], threshold: float) -> tuple[dict[str, str], dict[str, str]]:
    heuristic = parse_prompt(prompt)
    resolved = dict(learned_predictions)
    sources = {field: "model" for field in learned_predictions}

    heuristic_values = {
        "instrument": normalize_heuristic_instrument(heuristic.instrument),
        "mood": heuristic.mood if heuristic.mood in {"happy", "sad", "dark", "calm", "energetic", "dramatic"} else "neutral",
        "genre": heuristic.genre if heuristic.genre in {"pop", "rock", "electronic", "classical", "jazz", "cinematic", "ambient", "folk"} else "other",
        "density_level": heuristic.density_level,
        "polyphony_level": heuristic.polyphony_level,
        "note_duration_level": heuristic.note_duration_level,
        "register": heuristic.register,
    }

    force_override = {
        "instrument": (
            heuristic_values["instrument"] != "other"
            and (
                confidences.get("instrument", 0.0) < 0.92
                or (learned_predictions["instrument"] == "drums" and heuristic_values["instrument"] != "drums")
            )
        ),
        "mood": (
            heuristic_values["mood"] != "neutral"
            and (
                confidences.get("mood", 0.0) < 0.90
                or ((learned_predictions["mood"] == "dramatic") != (heuristic_values["mood"] == "dramatic"))
            )
        ),
        "genre": (
            heuristic_values["genre"] != "other"
            and (
                confidences.get("genre", 0.0) < 0.80
                or (heuristic_values["genre"] == "cinematic" and learned_predictions["genre"] != "cinematic")
            )
        ),
        "density_level": (
            heuristic_values["density_level"] != "medium"
            or confidences.get("density_level", 0.0) < max(threshold, 0.65)
        ),
        "polyphony_level": (
            heuristic_values["polyphony_level"] != "medium" and confidences.get("polyphony_level", 0.0) < 0.90
        ) or confidences.get("polyphony_level", 0.0) < max(threshold, 0.75),
        "note_duration_level": (
            heuristic_values["note_duration_level"] != "medium"
            or confidences.get("note_duration_level", 0.0) < max(threshold, 0.70)
        ),
        "register": (
            heuristic_values["register"] != "medium"
            or confidences.get("register", 0.0) < max(threshold, 0.75)
        ),
    }

    for field, should_override in force_override.items():
        if should_override:
            resolved[field] = heuristic_values[field]
            sources[field] = "heuristic_override"
        elif confidences.get(field, 0.0) < threshold:
            resolved[field] = heuristic_values[field]
            sources[field] = "heuristic_fallback"

    # Keep sparse / solo prompts from defaulting to unrealistically dense, highly polyphonic settings.
    if resolved["density_level"] == "low" and heuristic_values["polyphony_level"] != "high":
        resolved["polyphony_level"] = heuristic_values["polyphony_level"]
        sources["polyphony_level"] = "heuristic_override"

    return resolved, sources


def build_status(predictions: dict[str, str], track_id: int = 0, selected_bars: list[bool] | None = None) -> dict:
    if selected_bars is None:
        selected_bars = [False, False, True, False]

    polyphony_limit = POLYPHONY_TO_LIMIT[predictions["polyphony_level"]]
    temperature = temperature_from_predictions(predictions)
    return {
        "tracks": [
            {
                "track_id": track_id,
                "temperature": temperature,
                "instrument": INSTRUMENT_TO_MIDIGPT[predictions["instrument"]],
                "density": DENSITY_TO_SCORE[predictions["density_level"]],
                "track_type": infer_track_type(predictions),
                "ignore": False,
                "selected_bars": selected_bars,
                "min_polyphony_q": "POLYPHONY_ANY",
                "max_polyphony_q": "POLYPHONY_ANY",
                "autoregressive": False,
                "polyphony_hard_limit": polyphony_limit,
            }
        ]
    }


def default_params(ckpt: str = "models/model.ckpt") -> dict:
    return {
        "tracks_per_step": 1,
        "bars_per_step": 1,
        "model_dim": 4,
        "percentage": 100,
        "batch_size": 1,
        "temperature": 1.0,
        "max_steps": 200,
        "polyphony_hard_limit": 6,
        "shuffle": True,
        "verbose": True,
        "ckpt": ckpt,
        "sampling_seed": -1,
        "mask_top_k": 0,
    }


def load_prompts(path: str | None, prompt: str | None) -> list[str]:
    if prompt:
        return [prompt]
    if not path:
        raise ValueError("Provide either --prompt or --prompts.")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "caption" in data:
        return [str(data["caption"])]
    if isinstance(data, list):
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                prompts.append(str(item["caption"]))
        return prompts
    raise ValueError("Unsupported prompt format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict controls and emit MIDI-GPT-compatible status/param JSON.")
    parser.add_argument("--model-dir", default="artifacts/control_adapter")
    parser.add_argument("--prompts", help="JSON prompt file.")
    parser.add_argument("--prompt", help="Single prompt string.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--ckpt", default="models/model.ckpt")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for hybrid fallback.")
    parser.add_argument("--disable-hybrid", action="store_true", help="Use raw model predictions only.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_adapter(args.model_dir, device)
    prompts = load_prompts(args.prompts, args.prompt)
    predictions = predict_controls(model, tokenizer, prompts, device)

    items = []
    for item in predictions:
        if args.disable_hybrid:
            resolved_predictions = item.predictions
            sources = {field: "model" for field in item.predictions}
        else:
            resolved_predictions, sources = resolve_predictions(
                item.prompt, item.predictions, item.confidences, args.threshold
            )
        items.append(
            {
                "prompt": item.prompt,
                "predictions": resolved_predictions,
                "confidences": item.confidences,
                "sources": sources,
                "status": build_status(resolved_predictions),
                "param": default_params(args.ckpt),
            }
        )

    payload = {
        "device": device,
        "items": items,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
