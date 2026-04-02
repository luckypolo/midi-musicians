from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


CONTROL_FIELDS = {
    "instrument": ["piano", "guitar", "bass", "strings", "brass", "woodwind", "organ", "synth", "drums", "other"],
    "mood": ["happy", "sad", "dark", "calm", "energetic", "dramatic", "neutral"],
    "genre": ["pop", "rock", "electronic", "classical", "jazz", "cinematic", "ambient", "folk", "other"],
    "density_level": ["low", "medium", "high"],
    "polyphony_level": ["low", "medium", "high"],
    "note_duration_level": ["short", "medium", "long"],
    "register": ["low", "medium", "high"],
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _pick_first_match(values: list[str], candidates: list[str], fallback: str) -> str:
    normalized = [_normalize_text(value) for value in values]
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        for value in normalized:
            if normalized_candidate in value or value in normalized_candidate:
                return candidate
    return fallback


def map_instrument(instrument_summary: list[str] | None) -> str:
    if not instrument_summary:
        return "other"

    lowered = " ".join(_normalize_text(item) for item in instrument_summary)
    rules = [
        ("piano", ["piano", "keyboard"]),
        ("guitar", ["guitar"]),
        ("bass", ["bass"]),
        ("strings", ["string", "violin", "viola", "cello", "contrabass"]),
        ("brass", ["trumpet", "trombone", "brass", "horn"]),
        ("woodwind", ["flute", "clarinet", "oboe", "sax", "bassoon"]),
        ("organ", ["organ"]),
        ("synth", ["synth"]),
        ("drums", ["drum", "percussion"]),
    ]
    for label, patterns in rules:
        if any(pattern in lowered for pattern in patterns):
            return label
    return "other"


def map_mood(moods: list[str] | None, caption: str) -> str:
    text = " ".join((moods or []) + [caption]).lower()
    if any(token in text for token in ["happy", "uplifting", "joyful", "bright"]):
        return "happy"
    if any(token in text for token in ["sad", "melancholic", "emotional", "romantic"]):
        return "sad"
    if any(token in text for token in ["dark", "tense", "ominous"]):
        return "dark"
    if any(token in text for token in ["calm", "relaxing", "meditative", "peaceful"]):
        return "calm"
    if any(token in text for token in ["energetic", "driving", "fast", "lively"]):
        return "energetic"
    if any(token in text for token in ["dramatic", "epic", "cinematic"]):
        return "dramatic"
    return "neutral"


def map_genre(genres: list[str] | None, caption: str) -> str:
    text = " ".join((genres or []) + [caption]).lower()
    for genre in CONTROL_FIELDS["genre"]:
        if genre != "other" and genre in text:
            return genre
    if "film" in text:
        return "cinematic"
    return "other"


def map_density(tempo: int | None, tempo_word: str | None, duration_word: str | None) -> str:
    if tempo is not None:
        if tempo < 90:
            return "low"
        if tempo > 125:
            return "high"
    combined = " ".join(filter(None, [tempo_word, duration_word])).lower()
    if any(word in combined for word in ["largo", "adagio", "slow"]):
        return "low"
    if any(word in combined for word in ["allegro", "presto", "fast"]):
        return "high"
    return "medium"


def map_polyphony(instrument_summary: list[str] | None) -> str:
    count = len(instrument_summary or [])
    if count <= 2:
        return "low"
    if count >= 5:
        return "high"
    return "medium"


def map_note_duration(tempo: int | None, tempo_word: str | None) -> str:
    if tempo is not None:
        if tempo >= 140:
            return "short"
        if tempo <= 80:
            return "long"
    text = (tempo_word or "").lower()
    if text in {"presto", "allegro"}:
        return "short"
    if text in {"largo", "adagio"}:
        return "long"
    return "medium"


def map_register(instrument_summary: list[str] | None, caption: str) -> str:
    text = " ".join((instrument_summary or []) + [caption]).lower()
    if any(token in text for token in ["contrabass", "bass", "cello", "low register", "deep"]):
        return "low"
    if any(token in text for token in ["piccolo", "flute", "high register", "bright", "airy"]):
        return "high"
    return "medium"


def build_training_example(row: dict[str, Any]) -> dict[str, Any]:
    caption = row["caption"]
    return {
        "caption": caption,
        "instrument": map_instrument(row.get("instrument_summary")),
        "mood": map_mood(row.get("mood"), caption),
        "genre": map_genre(row.get("genre"), caption),
        "density_level": map_density(row.get("tempo"), row.get("tempo_word"), row.get("duration_word")),
        "polyphony_level": map_polyphony(row.get("instrument_summary")),
        "note_duration_level": map_note_duration(row.get("tempo"), row.get("tempo_word")),
        "register": map_register(row.get("instrument_summary"), caption),
        "source_location": row.get("location"),
        "tempo": row.get("tempo"),
        "key": row.get("key"),
        "time_signature": row.get("time_signature"),
        "test_set": bool(row.get("test_set", False)),
    }


def label_maps() -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    forward = {field: {label: index for index, label in enumerate(labels)} for field, labels in CONTROL_FIELDS.items()}
    reverse = {field: {index: label for index, label in enumerate(labels)} for field, labels in CONTROL_FIELDS.items()}
    return forward, reverse


class ControlAdapterModel(nn.Module):
    def __init__(self, encoder_name: str = "distilbert-base-uncased", dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(hidden_size, len(labels)) for field, labels in CONTROL_FIELDS.items()}
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        if hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = outputs[0][:, 0]
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.heads.items()}


def save_adapter(output_dir: str | Path, model: ControlAdapterModel, encoder_name: str, metadata: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "adapter_model.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps({"encoder_name": encoder_name, "control_fields": CONTROL_FIELDS, **metadata}, indent=2),
        encoding="utf-8",
    )


def load_adapter(model_dir: str | Path, device: str) -> tuple[ControlAdapterModel, Any, dict[str, dict[int, str]]]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = ControlAdapterModel(encoder_name=metadata["encoder_name"])
    state = torch.load(model_dir / "adapter_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(metadata["encoder_name"])
    _, reverse_maps = label_maps()
    return model, tokenizer, reverse_maps


@dataclass
class PredictionResult:
    prompt: str
    predictions: dict[str, str]
    confidences: dict[str, float]


def predict_controls(model: ControlAdapterModel, tokenizer: Any, prompts: list[str], device: str) -> list[PredictionResult]:
    _, reverse_maps = label_maps()
    encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded)

    results: list[PredictionResult] = []
    for row_index, prompt in enumerate(prompts):
        prediction_map: dict[str, str] = {}
        confidence_map: dict[str, float] = {}
        for field, field_logits in logits.items():
            probs = torch.softmax(field_logits[row_index], dim=-1)
            index = int(torch.argmax(probs).item())
            prediction_map[field] = reverse_maps[field][index]
            confidence_map[field] = float(probs[index].item())
        results.append(PredictionResult(prompt=prompt, predictions=prediction_map, confidences=confidence_map))
    return results
