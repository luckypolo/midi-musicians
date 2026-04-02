from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_python(python_exe: Path, script: Path, *args: str) -> None:
    cmd = [str(python_exe), str(script), *args]
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stem_to_index(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def build_summary(prompts: list[dict[str, Any]], payload: dict[str, Any], run_report: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    feature_by_index = {
        stem_to_index(Path(row["path"]).stem): row
        for row in features.get("files", [])
    }

    items = []
    for idx, prompt in enumerate(prompts):
        payload_item = payload["items"][idx]
        run_item = run_report["results"][idx]
        feature_row = feature_by_index.get(idx)
        items.append(
            {
                "id": prompt["id"],
                "prompt": prompt["caption"],
                "predictions": payload_item["predictions"],
                "sources": payload_item.get("sources", {}),
                "return_code": run_item["return_code"],
                "output_midi": run_item["output_midi"],
                "features": feature_row,
            }
        )

    by_field: dict[str, dict[str, list[dict[str, Any]]]] = {}
    tracked_fields = ["mood", "genre", "instrument", "density_level", "note_duration_level", "register"]
    for field in tracked_fields:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            label = item["predictions"][field]
            grouped.setdefault(label, []).append(item)
        by_field[field] = grouped

    return {
        "count": len(items),
        "summary": features.get("summary", {}),
        "items": items,
        "grouped_predictions": by_field,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid prompt-to-MIDI-GPT benchmark end to end.")
    parser.add_argument("--python", default=".venv/Scripts/python.exe")
    parser.add_argument("--prompts", default="prompts_benchmark.json")
    parser.add_argument("--model-dir", default="artifacts/control_adapter_augmented_v1")
    parser.add_argument("--threshold", default="0.65")
    parser.add_argument("--output-dir", default="outputs/benchmark_hybrid")
    parser.add_argument("--input-midi", default="outputs/text2midi_smoke/prompt_00.mid")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    python_exe = (repo_root / args.python).resolve()
    prompts_path = (repo_root / args.prompts).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_path = output_dir / "payload.json"
    run_dir = output_dir / "generated"
    feature_path = output_dir / "feature_report.json"
    summary_path = output_dir / "benchmark_summary.json"

    run_python(
        python_exe,
        repo_root / "midigpt_bridge.py",
        "--model-dir",
        str((repo_root / args.model_dir).resolve()),
        "--prompts",
        str(prompts_path),
        "--threshold",
        str(args.threshold),
        "--output",
        str(payload_path),
    )

    run_python(
        python_exe,
        repo_root / "run_midigpt_bridge_experiment.py",
        "--payload",
        str(payload_path),
        "--limit",
        str(len(load_json(prompts_path))),
        "--input-midi",
        str((repo_root / args.input_midi).resolve()),
        "--output-dir",
        str(run_dir),
    )

    run_python(
        python_exe,
        repo_root / "midi_analyzer.py",
        str(run_dir),
        "--output",
        str(feature_path),
    )

    prompts = load_json(prompts_path)
    payload = load_json(payload_path)
    run_report = load_json(run_dir / "run_report.json")
    features = load_json(feature_path)
    summary = build_summary(prompts, payload, run_report, features)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
