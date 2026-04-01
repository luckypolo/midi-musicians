from typing import Dict, Any
from resolve_conflict import determine_bars, resolve_proto_instrument, resolve_track_type, resolve_temperature, \
    TRACK_TYPE_STANDARD_DRUM

GENRE_TO_PROTO = {
    "any": "GENRE_MUSICMAP_ANY",
    "ambient": "GENRE_MUSICMAP_AMBIENT",
    "blues": "GENRE_MUSICMAP_BLUES",
    "classical": "GENRE_MUSICMAP_CLASSICAL",
    "country": "GENRE_MUSICMAP_COUNTRY",
    "folk": "GENRE_MUSICMAP_FOLK",
    "hip_hop": "GENRE_MUSICMAP_HIP_HOP",
    "house": "GENRE_MUSICMAP_HOUSE",
    "jazz": "GENRE_MUSICMAP_JAZZ",
    "latin": "GENRE_MUSICMAP_LATIN",
    "pop": "GENRE_MUSICMAP_POP",
    "reggae": "GENRE_MUSICMAP_REGGAE",
    "rock": "GENRE_MUSICMAP_CLASSIC_ROCK",
    "techno": "GENRE_MUSICMAP_TECHNO",
    "trance": "GENRE_MUSICMAP_TRANCE",
    "world": "GENRE_MUSICMAP_WORLD",
}

POLYPHONY_TO_PROTO = {
    "very_low": ("POLYPHONY_ONE", "POLYPHONY_ONE", 1),
    "low": ("POLYPHONY_ONE", "POLYPHONY_TWO", 2),
    "medium": ("POLYPHONY_TWO", "POLYPHONY_FOUR", 4),
    "high": ("POLYPHONY_THREE", "POLYPHONY_SIX", 6),
}

DENSITY_TO_PROTO = {
    "very_low": "DENSITY_TWO",
    "low": "DENSITY_FOUR",
    "medium": "DENSITY_SIX",
    "high": "DENSITY_EIGHT",
    "very_high": "DENSITY_TEN",
}

NOTE_DURATION_TO_PROTO = {
    "short": ("DURATION_SIXTEENTH", "DURATION_EIGHTH"),
    "medium": ("DURATION_EIGHTH", "DURATION_QUARTER"),
    "long": ("DURATION_QUARTER", "DURATION_HALF"),
}

REGISTER_TO_PITCH = {
    "low": (36, 60),
    "mid": (48, 72),
    "mid_high": (60, 84),
    "high": (72, 96),
}

DENSITY_TO_SILENCE = {
    "very_low": ("SILENCE_PROPORTION_LEVEL_SEVEN", "SILENCE_PROPORTION_LEVEL_TEN"),
    "low": ("SILENCE_PROPORTION_LEVEL_SIX", "SILENCE_PROPORTION_LEVEL_EIGHT"),
    "medium": ("SILENCE_PROPORTION_LEVEL_THREE", "SILENCE_PROPORTION_LEVEL_SIX"),
    "high": ("SILENCE_PROPORTION_LEVEL_TWO", "SILENCE_PROPORTION_LEVEL_FOUR"),
    "very_high": ("SILENCE_PROPORTION_LEVEL_ONE", "SILENCE_PROPORTION_LEVEL_THREE"),
}


def build_valid_status(piece_json: Dict[str, Any], gen_config: Dict[str, str], generation_mode: str) -> Dict[str, Any]:
    if not piece_json.get("tracks"):
        raise ValueError("Input piece does not contain any track.")

    # Select the first track as a target for generation
    target_track = piece_json["tracks"][0]

    # Determine number of bars
    num_bars = len(target_track.get("bars", []))

    # Which bars to generate and if generation is autoregressive
    selected_bars, autoregressive = determine_bars(num_bars, generation_mode)

    # Resolve parameters conflicts
    proto_instrument = resolve_proto_instrument(gen_config["instrument"])
    track_type = resolve_track_type(gen_config["instrument"])
    temperature = resolve_temperature(gen_config)

    status_track: Dict[str, Any] = {
        "track_id": 0,
        "track_type": track_type,
        "instrument": proto_instrument,
        "selected_bars": selected_bars,
        "autoregressive": autoregressive,
        "ignore": False,
        "temperature": temperature,
        "genre": GENRE_TO_PROTO[gen_config["genre"]],
    }

    if track_type == TRACK_TYPE_STANDARD_DRUM:
        status_track["density"] = DENSITY_TO_PROTO[gen_config["density_level"]]
    else:
        min_poly, max_poly, hard_limit = POLYPHONY_TO_PROTO[gen_config["polyphony_level"]]
        min_dur, max_dur = NOTE_DURATION_TO_PROTO[gen_config["note_duration_level"]]
        min_pitch, max_pitch = REGISTER_TO_PITCH[gen_config["register"]]
        silence_min, silence_max = DENSITY_TO_SILENCE[gen_config["density_level"]]

        status_track.update(
            {
                "min_polyphony_q": min_poly,
                "max_polyphony_q": max_poly,
                "min_note_duration_q": min_dur,
                "max_note_duration_q": max_dur,
                "min_pitch": min_pitch,
                "max_pitch": max_pitch,
                "silence_proportion_min": silence_min,
                "silence_proportion_max": silence_max,
                "polyphony_hard_limit": hard_limit,
            }
        )

    return {
        "tracks": [status_track],
        "decode_final": True,
        "full_resolution": False,
    }


def build_parami(ckpt: str, generation_mode: str) -> Dict[str, Any]:
    return {
        "tracks_per_step": 1,
        "bars_per_step": 1,
        "model_dim": 4,
        "percentage": 100,
        "batch_size": 1,
        "temperature": 1.0,
        "use_per_track_temperature": True,
        "max_steps": 200,
        "polyphony_hard_limit": 6,
        "shuffle": generation_mode != "autoregressive",
        "verbose": True,
        "ckpt": ckpt,
        "sampling_seed": -1,
        "mask_top_k": 0.0,
    }
