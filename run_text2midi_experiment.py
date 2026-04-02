from __future__ import annotations

import argparse
import json
from pathlib import Path

from midi_analyzer import extract_features
from prompt_to_controls import load_prompts as load_prompt_strings
from prompt_to_controls import parse_prompt
from text2midi_runner import generate_midis, load_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full Text2midi experiment: controls -> generation -> feature analysis.")
    parser.add_argument("--prompts", default="prompts.json", help="Prompt JSON file.")
    parser.add_argument("--output-dir", default="outputs/experiment_text2midi", help="Experiment output directory.")
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    prompt_path = (repo_root / args.prompts).resolve() if not Path(args.prompts).is_absolute() else Path(args.prompts)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_texts = load_prompt_strings(prompt_path)
    controls = [parse_prompt(prompt) for prompt in prompt_texts]
    prompts = load_prompts(str(prompt_path))
    manifest = generate_midis(prompts, output_dir / "generated_midis", max_len=args.max_len, temperature=args.temperature)

    analyses = []
    for output in manifest["outputs"]:
        features = extract_features(output["midi_path"])
        analyses.append(
            {
                "id": output["id"],
                "caption": output["caption"],
                "controls": next((control.__dict__ for control in controls if control.prompt == output["caption"]), None),
                "features": features.__dict__,
            }
        )

    report = {
        "manifest": manifest,
        "comparisons": analyses,
    }
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
