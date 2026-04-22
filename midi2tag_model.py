from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from miditok import REMI, TokenizerConfig

from control_adapter import CONTROL_FIELDS


MIDI2TAG_FIELDS = [
    "mood",
    "density_level",
    "polyphony_level",
    "note_duration_level",
    "register",
]


def midi2tag_label_maps(
    fields: list[str] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    active_fields = fields or MIDI2TAG_FIELDS
    forward = {field: {label: index for index, label in enumerate(CONTROL_FIELDS[field])} for field in active_fields}
    reverse = {field: {index: label for index, label in enumerate(CONTROL_FIELDS[field])} for field in active_fields}
    return forward, reverse


def build_remi_tokenizer() -> REMI:
    config = TokenizerConfig(use_programs=False)
    return REMI(config)


def tokenizer_vocab_size(tokenizer: REMI) -> int:
    return max(int(index) for index in tokenizer.vocab.values()) + 1


def collect_midi_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))


def tokenize_midi(
    midi_path: str | Path,
    tokenizer: REMI | None = None,
    track_policy: str = "single",
    max_seq_len: int | None = None,
) -> list[int]:
    tokenizer = tokenizer or build_remi_tokenizer()
    sequences = tokenizer(Path(midi_path))
    if not isinstance(sequences, list):
        sequences = [sequences]
    sequences = [sequence for sequence in sequences if getattr(sequence, "ids", None)]
    if not sequences:
        raise ValueError(f"No token sequence produced for {midi_path}")

    if track_policy == "single":
        if len(sequences) != 1:
            raise ValueError(f"Expected single-track MIDI, got {len(sequences)} tracks for {midi_path}")
        token_ids = list(sequences[0].ids)
    elif track_policy == "longest":
        token_ids = list(max(sequences, key=lambda sequence: len(sequence.ids)).ids)
    elif track_policy == "concat":
        token_ids = [token_id for sequence in sequences for token_id in sequence.ids]
    else:
        raise ValueError(f"Unsupported track_policy: {track_policy}")

    if max_seq_len:
        token_ids = token_ids[:max_seq_len]
    if not token_ids:
        raise ValueError(f"Empty token sequence for {midi_path}")
    return token_ids


@dataclass
class Midi2TagPrediction:
    midi_path: str
    predictions: dict[str, str]
    confidences: dict[str, float]
    caption: str


class Midi2TagModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        fields: list[str] | None = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.fields = fields or MIDI2TAG_FIELDS
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, len(CONTROL_FIELDS[field])) for field in self.fields}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        encoded = self.encoder(pooled)
        return {field: head(encoded) for field, head in self.heads.items()}


def make_caption_from_tags(predictions: dict[str, str]) -> str:
    mood = predictions.get("mood", "neutral")
    density = predictions.get("density_level", "medium")
    polyphony = predictions.get("polyphony_level", "medium")
    duration = predictions.get("note_duration_level", "medium")
    register = predictions.get("register", "medium")
    mood_text = "" if mood == "neutral" else f"{mood} "
    return (
        f"A {mood_text}MIDI passage with {density} density, {polyphony} polyphony, "
        f"{duration} notes, and a {register} register."
    )


def save_midi2tag_model(
    output_dir: str | Path,
    model: Midi2TagModel,
    metadata: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "midi2tag_model.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_midi2tag_model(model_dir: str | Path, device: str) -> tuple[Midi2TagModel, dict[str, Any]]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = Midi2TagModel(
        vocab_size=int(metadata["vocab_size"]),
        fields=list(metadata["fields"]),
        embedding_dim=int(metadata["embedding_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        dropout=float(metadata.get("dropout", 0.0)),
        pad_id=int(metadata.get("pad_id", 0)),
    )
    state = torch.load(model_dir / "midi2tag_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, metadata
