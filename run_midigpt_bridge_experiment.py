from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_midigpt(build_dir: Path, torch_lib_dir: Path):
    os.add_dll_directory(str(build_dir))
    os.add_dll_directory(str(torch_lib_dir))
    sys.path.insert(0, str(build_dir))
    import midigpt  # type: ignore

    return midigpt


def resolve_checkpoint(ckpt_arg: str, repo_root: Path) -> Path:
    ckpt_path = Path(ckpt_arg)
    if ckpt_path.is_absolute():
        return ckpt_path
    return (repo_root / ckpt_path).resolve()


def is_drum_track(track: dict) -> bool:
    return str(track.get("trackType", "")).upper().endswith("DRUM_TRACK")


def apply_requested_controls(native_status: dict, requested: dict) -> dict:
    status = json.loads(json.dumps(native_status))
    selected_bars = requested["selected_bars"]
    desired_instrument = requested["instrument"]
    requested_density = requested["density"]
    requested_polyphony_limit = requested["polyphony_hard_limit"]

    non_drum_tracks = [track for track in status["tracks"] if not is_drum_track(track)]
    drum_tracks = [track for track in status["tracks"] if is_drum_track(track)]

    # Actuate sparse prompts more aggressively by reducing active accompaniment tracks.
    max_active_tracks = len(non_drum_tracks)
    if requested_density <= 3 or requested_polyphony_limit <= 2:
        max_active_tracks = min(max_active_tracks, 2)
    elif requested_density <= 6 or requested_polyphony_limit <= 6:
        max_active_tracks = min(max_active_tracks, 4)

    for index, track in enumerate(non_drum_tracks):
        track["temperature"] = requested["temperature"]
        track["density"] = requested_density
        track["polyphonyHardLimit"] = requested_polyphony_limit
        is_active = index < max_active_tracks
        track["ignore"] = not is_active
        track["selectedBars"] = selected_bars if is_active else [False for _ in selected_bars]

    if requested["track_type"] == "DRUM_TRACK":
        for track in drum_tracks:
            track["temperature"] = requested["temperature"]
            track["selectedBars"] = selected_bars
            track["polyphonyHardLimit"] = requested_polyphony_limit
    else:
        if non_drum_tracks:
            non_drum_tracks[0]["instrument"] = desired_instrument

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIDI-GPT sampling from bridge payload JSON.")
    parser.add_argument("--payload", default="outputs/midigpt_bridge_payload.json")
    parser.add_argument("--input-midi", default="outputs/text2midi_smoke/prompt_00.mid")
    parser.add_argument(
        "--checkpoint",
        default="vendor/MIDI-GPT/models/unzipped/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt",
    )
    parser.add_argument("--output-dir", default="outputs/midigpt_bridge_run")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1, help="How many payload items to run.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    build_dir = repo_root / "vendor" / "MIDI-GPT" / "build-ninja"
    torch_lib_dir = repo_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    ckpt_path = resolve_checkpoint(args.checkpoint, repo_root)
    payload_path = repo_root / args.payload
    input_midi = repo_root / args.input_midi
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    items = payload["items"][: args.limit]

    midigpt = load_midigpt(build_dir, torch_lib_dir)
    encoder = midigpt.ExpressiveEncoder()
    callbacks = midigpt.CallbackManager()
    piece_json = encoder.midi_to_json(str(input_midi))
    native_status = json.loads(midigpt.status_from_piece(piece_json))

    results = []
    for idx, item in enumerate(items):
        requested = item["status"]["tracks"][0]
        status = apply_requested_controls(native_status, requested)
        param = item["param"]
        param["ckpt"] = str(ckpt_path)
        out_midi = output_dir / f"bridge_run_{idx:02d}.mid"

        midi_json_str, return_code = midigpt.sample_multi_step(
            piece_json,
            json.dumps(status),
            json.dumps(param),
            args.max_attempts,
            callbacks,
        )
        if isinstance(midi_json_str, (list, tuple)):
            midi_json_str = midi_json_str[0]
        encoder.json_to_midi(midi_json_str, str(out_midi))
        results.append(
            {
                "prompt": item["prompt"],
                "input_midi": str(input_midi),
                "output_midi": str(out_midi),
                "return_code": int(return_code),
                "status": status,
                "param": param,
            }
        )

    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
