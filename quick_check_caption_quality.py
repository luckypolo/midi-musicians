from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ATTRIBUTE_PATTERNS = {
    "mood": re.compile(r"\b(happy|sad|dark|calm|energetic|uplifting|melancholic|relaxing|dramatic|epic)\b", re.I),
    "genre": re.compile(r"\b(pop|rock|electronic|classical|jazz|trance|cinematic|ambient|folk|metal)\b", re.I),
    "instrument": re.compile(r"\b(piano|guitar|bass|drums|violin|cello|saxophone|flute|organ|synth)\b", re.I),
    "tempo": re.compile(r"\b(\d{2,3}\s?bpm|allegro|adagio|presto|moderato|largo)\b", re.I),
    "key": re.compile(r"\bkey of [A-G](#|b)?\s?(major|minor)\b", re.I),
    "time_signature": re.compile(r"\b\d/\d\b"),
}


@dataclass
class CaptionCheck:
    caption: str
    length_chars: int
    length_words: int
    detected_fields: dict[str, bool]
    score: int


def score_caption(caption: str) -> CaptionCheck:
    detected = {name: bool(pattern.search(caption)) for name, pattern in ATTRIBUTE_PATTERNS.items()}
    score = sum(1 for value in detected.values() if value)
    return CaptionCheck(
        caption=caption,
        length_chars=len(caption),
        length_words=len(caption.split()),
        detected_fields=detected,
        score=score,
    )


def load_captions(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            captions = []
            for item in data:
                if isinstance(item, str):
                    captions.append(item)
                elif isinstance(item, dict) and "caption" in item:
                    captions.append(str(item["caption"]))
            return captions
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick heuristic check for caption richness.")
    parser.add_argument("input", help="Path to a txt/json file containing captions.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    captions = load_captions(Path(args.input))
    checks = [score_caption(caption) for caption in captions]

    payload = {
        "caption_count": len(checks),
        "average_score": (sum(check.score for check in checks) / len(checks)) if checks else 0.0,
        "captions": [asdict(check) for check in checks],
    }

    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
