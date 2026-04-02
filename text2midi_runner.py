from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer

REPO_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "Text2midi"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from model.transformer_model import Transformer  # noqa: E402


DEFAULT_PROMPTS = [
    {
        "id": "happy_pop",
        "caption": "A cheerful pop song with piano, acoustic guitar, bass, and drums in C major, medium tempo, and a bright uplifting mood.",
    },
    {
        "id": "dark_cinematic",
        "caption": "A dark cinematic piece with cello, contrabass, and percussion in A minor, moderate tempo, with a dramatic and tense atmosphere.",
    },
]


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_PROMPTS

    prompt_path = Path(path)
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "caption" in data:
            data = [data]
        elif "prompts" in data and isinstance(data["prompts"], list):
            data = data["prompts"]
        else:
            raise ValueError("Prompt file must be a list of prompts, a single prompt object, or a {\"prompts\": [...]} object.")
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            prompts.append({"id": f"prompt_{index:02d}", "caption": item})
        elif isinstance(item, dict):
            prompts.append(
                {
                    "id": str(item.get("id", f"prompt_{index:02d}")),
                    "caption": str(item["caption"]),
                }
            )
    return prompts


def load_model(device: str) -> tuple[Transformer, T5Tokenizer, object]:
    repo_id = "amaai-lab/text2midi"
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

    with open(tokenizer_path, "rb") as handle:
        remi_tokenizer = pickle.load(handle)

    model = Transformer(len(remi_tokenizer), 768, 8, 2048, 18, 1024, False, 8, device=device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return model, text_tokenizer, remi_tokenizer


def generate_midis(
    prompts: list[dict[str, str]],
    output_dir: Path,
    max_len: int = 1400,
    temperature: float = 0.9,
) -> dict[str, object]:
    device = resolve_device()
    model, text_tokenizer, remi_tokenizer = load_model(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for prompt in prompts:
        encoded = text_tokenizer(prompt["caption"], return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model.generate(input_ids, attention_mask, max_len=max_len, temperature=temperature)

        midi = remi_tokenizer.decode(output[0].tolist())
        output_path = output_dir / f"{prompt['id']}.mid"
        midi.dump_midi(str(output_path))

        generated.append(
            {
                "id": prompt["id"],
                "caption": prompt["caption"],
                "midi_path": str(output_path),
            }
        )

    manifest = {
        "device": device,
        "repo_id": "amaai-lab/text2midi",
        "temperature": temperature,
        "max_len": max_len,
        "outputs": generated,
    }
    (output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI files from text prompts using Text2midi.")
    parser.add_argument("--prompts", help="Optional JSON file with prompts. Defaults to a built-in small set.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "text2midi"), help="Output directory.")
    parser.add_argument("--max-len", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    manifest = generate_midis(prompts, Path(args.output_dir), max_len=args.max_len, temperature=args.temperature)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
