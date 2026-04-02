from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from control_adapter import load_adapter, predict_controls


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
    raise ValueError("Unsupported prompt file format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained control adapter.")
    parser.add_argument("--model-dir", default="artifacts/control_adapter")
    parser.add_argument("--prompts", help="JSON prompt file.")
    parser.add_argument("--prompt", help="Single prompt string.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_adapter(args.model_dir, device)
    prompts = load_prompts(args.prompts, args.prompt)
    results = predict_controls(model, tokenizer, prompts, device)
    payload = {
        "device": device,
        "results": [
            {
                "prompt": result.prompt,
                "predictions": result.predictions,
                "confidences": result.confidences,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
