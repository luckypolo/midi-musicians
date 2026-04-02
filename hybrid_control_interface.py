from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from control_adapter import load_adapter, predict_controls
from prompt_to_controls import parse_prompt


def build_hybrid_result(prompt: str, learned_predictions: dict[str, str], confidences: dict[str, float], threshold: float) -> dict:
    heuristic = parse_prompt(prompt).__dict__
    predictions: dict[str, str] = {}
    sources: dict[str, str] = {}

    for field, heuristic_value in heuristic.items():
        if field == "prompt":
            continue
        if field in learned_predictions and confidences.get(field, 0.0) >= threshold:
            predictions[field] = learned_predictions[field]
            sources[field] = "model"
        elif field in learned_predictions:
            predictions[field] = str(heuristic_value)
            sources[field] = "heuristic_fallback"
        else:
            predictions[field] = str(heuristic_value)
            sources[field] = "heuristic_only"

    return {
        "prompt": prompt,
        "predictions": predictions,
        "confidences": confidences,
        "sources": sources,
    }


def load_prompts(path: str | None, prompt: str | None) -> list[str]:
    if prompt:
        return [prompt]
    if not path:
        raise ValueError("Provide either --prompt or --prompts.")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "caption" in data:
        return [str(data["caption"])]
    prompts: list[str] = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            prompts.append(str(item["caption"]))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid natural-language control interface with learned predictions and heuristic fallback.")
    parser.add_argument("--model-dir", default="artifacts/control_adapter")
    parser.add_argument("--prompts", help="JSON prompt file.")
    parser.add_argument("--prompt", help="Single prompt string.")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_adapter(args.model_dir, device)
    prompts = load_prompts(args.prompts, args.prompt)
    model_results = predict_controls(model, tokenizer, prompts, device)

    payload = {
        "device": device,
        "threshold": args.threshold,
        "results": [
            build_hybrid_result(result.prompt, result.predictions, result.confidences, args.threshold)
            for result in model_results
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
