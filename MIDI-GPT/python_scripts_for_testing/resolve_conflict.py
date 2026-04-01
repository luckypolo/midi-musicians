from typing import Dict, Tuple, List

from utils import clamp

TRACK_TYPE_STANDARD = 10
TRACK_TYPE_STANDARD_DRUM = 11

POLYPHONY_ORDER = ["very_low", "low", "medium", "high"]
DENSITY_ORDER = ["very_low", "low", "medium", "high", "very_high"]
DURATION_ORDER = ["short", "medium", "long"]
REGISTER_ORDER = ["low", "mid", "mid_high", "high"]


def determine_bars(num_bars: int, generation_mode: str) -> Tuple[List[bool], bool]:
    if num_bars <= 0:
        raise ValueError("Piece must contain at least one bar on the target track.")

    if generation_mode == "autoregressive":
        return [True] * num_bars, True

    if generation_mode == "full_conditional":
        return [True] * num_bars, False

    # Default: keep first bar as context and generate the rest.
    if num_bars == 1:
        return [True], False
    return [False] + [True] * (num_bars - 1), False


def resolve_track_type(instrument: str) -> int:
    return TRACK_TYPE_STANDARD_DRUM if instrument == "drums" else TRACK_TYPE_STANDARD


def resolve_proto_instrument(instrument: str) -> str:
    # Keep family instruments when they are more suitable than a single patch.
    if instrument in {"piano", "guitar", "strings", "drums"}:
        return instrument
    return instrument


def resolve_temperature(gen_config: Dict[str, str]) -> float:
    temp = 0.85

    if gen_config["mood"] == "calm":
        temp -= 0.10
    elif gen_config["mood"] == "sad":
        temp -= 0.05
    elif gen_config["mood"] == "happy":
        temp += 0.05
    elif gen_config["mood"] == "energetic":
        temp += 0.10
    elif gen_config["mood"] == "dark":
        temp -= 0.02

    if gen_config["complexity"] == "simple":
        temp -= 0.05
    elif gen_config["complexity"] == "complex":
        temp += 0.05

    return round(clamp(temp, 0.5, 1.2), 2)
