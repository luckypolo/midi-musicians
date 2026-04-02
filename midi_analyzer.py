from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pretty_midi


@dataclass
class MidiFeatures:
    path: str
    instruments: int
    drum_tracks: int
    pitched_tracks: int
    note_count: int
    duration_seconds: float
    notes_per_second: float
    avg_note_duration: float
    avg_velocity: float
    pitch_min: int | None
    pitch_max: int | None
    pitch_span: int | None
    avg_pitch: float | None
    estimated_tempo: float
    max_polyphony: int


def _max_polyphony(notes: list[pretty_midi.Note]) -> int:
    if not notes:
        return 0

    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((note.start, 1))
        events.append((note.end, -1))

    # End events go first at tied timestamps so a note ending and another starting
    # at the same time does not inflate overlap.
    events.sort(key=lambda item: (item[0], item[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def extract_features(midi_path: str | Path) -> MidiFeatures:
    midi_path = Path(midi_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    notes = [note for instrument in midi.instruments for note in instrument.notes]
    pitched_notes = [note for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes]

    duration = float(midi.get_end_time() or 0.0)
    note_durations = [note.end - note.start for note in notes]
    velocities = [note.velocity for note in notes]
    pitches = [note.pitch for note in pitched_notes]

    pitch_min = min(pitches) if pitches else None
    pitch_max = max(pitches) if pitches else None
    pitch_span = (pitch_max - pitch_min) if pitches else None
    avg_pitch = statistics.fmean(pitches) if pitches else None

    try:
        estimated_tempo = float(midi.estimate_tempo())
    except Exception:
        estimated_tempo = 0.0

    return MidiFeatures(
        path=str(midi_path),
        instruments=len(midi.instruments),
        drum_tracks=sum(1 for instrument in midi.instruments if instrument.is_drum),
        pitched_tracks=sum(1 for instrument in midi.instruments if not instrument.is_drum),
        note_count=len(notes),
        duration_seconds=duration,
        notes_per_second=(len(notes) / duration) if duration > 0 else 0.0,
        avg_note_duration=statistics.fmean(note_durations) if note_durations else 0.0,
        avg_velocity=statistics.fmean(velocities) if velocities else 0.0,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_span=pitch_span,
        avg_pitch=avg_pitch,
        estimated_tempo=estimated_tempo,
        max_polyphony=_max_polyphony(notes),
    )


def collect_midis(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.mid")) + sorted(path.rglob("*.midi"))


def summarize(features: Iterable[MidiFeatures]) -> dict[str, float | int]:
    items = list(features)
    if not items:
        return {"files": 0}

    def avg(field: str) -> float:
        values = [getattr(item, field) for item in items]
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        return float(statistics.fmean(numeric_values)) if numeric_values else 0.0

    return {
        "files": len(items),
        "avg_note_count": avg("note_count"),
        "avg_duration_seconds": avg("duration_seconds"),
        "avg_notes_per_second": avg("notes_per_second"),
        "avg_note_duration": avg("avg_note_duration"),
        "avg_velocity": avg("avg_velocity"),
        "avg_estimated_tempo": avg("estimated_tempo"),
        "avg_max_polyphony": avg("max_polyphony"),
        "avg_pitch_span": avg("pitch_span"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MIDI files and extract lightweight musical features.")
    parser.add_argument("path", help="Path to a MIDI file or directory of MIDI files.")
    parser.add_argument("--output", help="Optional JSON file for the per-file feature dump.")
    parser.add_argument("--summary-only", action="store_true", help="Only print an aggregated summary.")
    args = parser.parse_args()

    midi_paths = collect_midis(Path(args.path))
    feature_rows = [extract_features(path) for path in midi_paths]

    payload = {
        "summary": summarize(feature_rows),
        "files": [] if args.summary_only else [asdict(row) for row in feature_rows],
    }

    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
