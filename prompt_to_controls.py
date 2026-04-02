from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


INSTRUMENTS = [
    "piano",
    "acoustic guitar",
    "electric guitar",
    "guitar",
    "bass",
    "drums",
    "violin",
    "viola",
    "cello",
    "contrabass",
    "flute",
    "saxophone",
    "organ",
    "synth",
    "percussion",
]

INSTRUMENT_FAMILIES = {
    "piano": ["piano", "keyboard"],
    "guitar": ["acoustic guitar", "electric guitar", "guitar"],
    "bass": ["bass", "contrabass"],
    "strings": ["violin", "viola", "cello", "string"],
    "brass": ["trumpet", "trombone", "horn", "brass"],
    "woodwind": ["flute", "saxophone", "clarinet", "oboe", "bassoon"],
    "organ": ["organ"],
    "synth": ["synth"],
    "drums": ["drums", "percussion"],
}

MOOD_MAP = {
    "happy": "happy",
    "uplifting": "happy",
    "bright": "happy",
    "sad": "sad",
    "melancholic": "sad",
    "dark": "dark",
    "tense": "dark",
    "calm": "calm",
    "relaxing": "calm",
    "energetic": "energetic",
    "dramatic": "dramatic",
}

GENRES = ["pop", "rock", "electronic", "classical", "jazz", "cinematic", "ambient", "trance", "folk"]


@dataclass
class PromptControls:
    prompt: str
    instrument: str
    mood: str
    complexity: str
    polyphony_level: str
    density_level: str
    register: str
    note_duration_level: str
    genre: str
    key: str | None
    tempo_hint: str | None
    time_signature: str | None


def first_match(options: list[str], text: str, default: str) -> str:
    lowered = text.lower()
    for option in options:
        if option in lowered:
            return option
    return default


def infer_primary_instrument(text: str) -> str:
    lowered = text.lower()
    scores = {family: 0 for family in INSTRUMENT_FAMILIES}
    for family, aliases in INSTRUMENT_FAMILIES.items():
        for alias in aliases:
            if alias in lowered:
                scores[family] += 1

    if "classical" in lowered or "orchestral" in lowered:
        scores["strings"] += 1
    if "cinematic" in lowered:
        scores["strings"] += 1
        scores["bass"] += 1
    if "pop" in lowered:
        scores["piano"] += 1
        scores["guitar"] += 1

    best_family, best_score = max(scores.items(), key=lambda item: item[1])
    return best_family if best_score > 0 else "piano"


def infer_mood(text: str) -> str:
    lowered = text.lower()
    for token, mood in MOOD_MAP.items():
        if token in lowered:
            return mood
    return "neutral"


def infer_complexity(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["minimal", "simple", "sparse"]):
        return "low"
    if any(token in lowered for token in ["dense", "complex", "layered", "intricate"]):
        return "high"
    return "medium"


def infer_density(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["slow", "sparse", "calm", "minimal"]):
        return "low"
    if any(token in lowered for token in ["fast", "energetic", "dense", "driving"]):
        return "high"
    return "medium"


def infer_polyphony(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["solo", "monophonic", "single-line"]):
        return "low"
    if any(token in lowered for token in ["orchestral", "layered", "rich", "polyphonic"]):
        return "high"
    return "medium"


def infer_register(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["deep", "low register", "bass-heavy", "dark"]):
        return "low"
    if any(token in lowered for token in ["bright", "high register", "sparkling", "airy"]):
        return "high"
    return "medium"


def infer_note_duration(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["staccato", "short", "plucky"]):
        return "short"
    if any(token in lowered for token in ["sustained", "legato", "long"]):
        return "long"
    return "medium"


def extract_key(text: str) -> str | None:
    match = re.search(r"\b([A-G](?:#|b)?)\s+(major|minor)\b", text, re.IGNORECASE)
    return f"{match.group(1).upper()} {match.group(2).lower()}" if match else None


def extract_tempo_hint(text: str) -> str | None:
    bpm = re.search(r"\b(\d{2,3})\s?bpm\b", text, re.IGNORECASE)
    if bpm:
        return f"{bpm.group(1)} BPM"
    for word in ["largo", "adagio", "moderato", "allegro", "presto", "slow", "medium tempo", "fast"]:
        if word in text.lower():
            return word
    return None


def extract_time_signature(text: str) -> str | None:
    match = re.search(r"\b(\d/\d)\b", text)
    return match.group(1) if match else None


def parse_prompt(prompt: str) -> PromptControls:
    return PromptControls(
        prompt=prompt,
        instrument=infer_primary_instrument(prompt),
        mood=infer_mood(prompt),
        complexity=infer_complexity(prompt),
        polyphony_level=infer_polyphony(prompt),
        density_level=infer_density(prompt),
        register=infer_register(prompt),
        note_duration_level=infer_note_duration(prompt),
        genre=first_match(GENRES, prompt, "unknown"),
        key=extract_key(prompt),
        tempo_hint=extract_tempo_hint(prompt),
        time_signature=extract_time_signature(prompt),
    )


def load_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "caption" in data:
            return [str(data["caption"])]
        if "prompts" in data:
            data = data["prompts"]

    prompts: list[str] = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            prompts.append(str(item["caption"]))
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert natural-language prompts into lightweight symbolic controls.")
    parser.add_argument("--prompts", required=True, help="JSON prompt file.")
    parser.add_argument("--output", help="Optional JSON output file.")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    controls = [asdict(parse_prompt(prompt)) for prompt in prompts]
    payload = {"count": len(controls), "controls": controls}
    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
