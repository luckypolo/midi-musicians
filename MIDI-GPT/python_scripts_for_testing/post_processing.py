from typing import Dict, Any

from utils import capped_level, raised_level, canonicalize_string
from resolve_conflict import POLYPHONY_ORDER, REGISTER_ORDER, DURATION_ORDER, DENSITY_ORDER

# Proto constants (from enum.proto / track_type.proto / midi.proto)

### SYNONYMES ###
INSTRUMENT_SYNONYMS = {
    "grand_piano": "acoustic_grand_piano",
    "piano": "piano",
    "acoustic_piano": "acoustic_grand_piano",
    "nylon_guitar": "acoustic_guitar_nylon",
    "acoustic_guitar": "acoustic_guitar_nylon",
    "guitar": "guitar",
    "strings": "strings",
    "string": "strings",
    "violin": "violin",
    "flute": "flute",
    "drum": "drums",
    "drums": "drums",
    "drum_kit": "drums",
    "standard_drum_kit": "drums",
}
SYNONYMS = {
    "instrument": INSTRUMENT_SYNONYMS
}

### ALLOWED VARIABLES ###
ALLOWED_INSTRUMENTS = {
    "acoustic_grand_piano",
    "piano",
    "acoustic_guitar_nylon",
    "guitar",
    "violin",
    "strings",
    "flute",
    "drums",
}
ALLOWED_MOODS = {"neutral", "happy", "sad", "calm", "energetic", "dark"}
ALLOWED_COMPLEXITIES = {"simple", "moderate", "complex"}
ALLOWED_POLYPHONY_LEVELS = {"very_low", "low", "medium", "high"}
ALLOWED_DENSITY_LEVELS = {"very_low", "low", "medium", "high", "very_high"}
ALLOWED_REGISTERS = {"low", "mid", "mid_high", "high"}
ALLOWED_NOTE_DURATION_LEVELS = {"short", "medium", "long"}
ALLOWED_GENRES = {
    "any",
    "ambient",
    "blues",
    "classical",
    "country",
    "folk",
    "hip_hop",
    "house",
    "jazz",
    "latin",
    "pop",
    "reggae",
    "rock",
    "techno",
    "trance",
    "world",
}
ALLOWED_VARIABLES = {
    "instrument": ALLOWED_INSTRUMENTS,
    "mood": ALLOWED_MOODS,
    "complexity": ALLOWED_COMPLEXITIES,
    "polyphony_level": ALLOWED_POLYPHONY_LEVELS,
    "density_level": ALLOWED_DENSITY_LEVELS,
    "register": ALLOWED_REGISTERS,
    "note_duration_level": ALLOWED_NOTE_DURATION_LEVELS,
    "genre": ALLOWED_GENRES
}


def normalize_generation_config(gen_config: Dict[str, str]) -> Dict[str, str]:
    normalized_config = dict(gen_config)

    if normalized_config["complexity"] == "simple":
        normalized_config["polyphony_level"] = capped_level(normalized_config["polyphony_level"], POLYPHONY_ORDER,
                                                            "low")
        normalized_config["density_level"] = capped_level(normalized_config["density_level"], DENSITY_ORDER, "medium")
        normalized_config["note_duration_level"] = raised_level(normalized_config["note_duration_level"],
                                                                DURATION_ORDER, "medium")

    if normalized_config["complexity"] == "complex":
        normalized_config["polyphony_level"] = raised_level(normalized_config["polyphony_level"], POLYPHONY_ORDER,
                                                            "medium")
        normalized_config["density_level"] = raised_level(normalized_config["density_level"], DENSITY_ORDER, "medium")

    if normalized_config["mood"] == "calm":
        normalized_config["density_level"] = capped_level(normalized_config["density_level"], DENSITY_ORDER, "medium")
        normalized_config["note_duration_level"] = raised_level(normalized_config["note_duration_level"],
                                                                DURATION_ORDER, "medium")

    if normalized_config["mood"] == "energetic":
        normalized_config["density_level"] = raised_level(normalized_config["density_level"], DENSITY_ORDER, "medium")
        normalized_config["note_duration_level"] = capped_level(normalized_config["note_duration_level"],
                                                                DURATION_ORDER, "medium")

    if normalized_config["mood"] == "happy":
        normalized_config["register"] = raised_level(normalized_config["register"], REGISTER_ORDER, "mid")

    if normalized_config["mood"] in {"sad", "dark"}:
        normalized_config["register"] = capped_level(normalized_config["register"], REGISTER_ORDER, "mid")

    if normalized_config["instrument"] == "drums":
        normalized_config["polyphony_level"] = "high"
        normalized_config["note_duration_level"] = "short"
        normalized_config["register"] = "mid"

    return normalized_config


DEFAULT_DSL = {
    "instrument": "acoustic_grand_piano",
    "mood": "neutral",
    "complexity": "moderate",
    "polyphony_level": "medium",
    "density_level": "medium",
    "register": "mid",
    "note_duration_level": "medium",
    "genre": "any",
}


def sanitize_raw_json(raw_controls: Dict[str, Any]) -> Dict[str, str]:
    controls = dict(DEFAULT_DSL)
    for key in controls:
        controls[key] = canonicalize_string(raw_controls.get(key), DEFAULT_DSL[key], SYNONYMS.get(key, {}),
                                            ALLOWED_VARIABLES[key])
    return normalize_generation_config(controls)
