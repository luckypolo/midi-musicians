import argparse
import json
import os
import sys
from typing import Any, Dict
from build_params import build_parami, build_valid_status
from post_processing import ALLOWED_VARIABLES, sanitize_raw_json

import requests

sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt


# LLM interface constrained DSL (prompt made by chatgpt)
def build_system_prompt() -> str:
    return """
You convert a natural-language music prompt into a small semantic control DSL for MIDIGPT.

Return JSON only. Do not explain anything.
Return exactly these fields:
{
  "instrument": string,
  "mood": string,
  "complexity": string,
  "polyphony_level": string,
  "density_level": string,
  "register": string,
  "note_duration_level": string,
  "genre": string
}

Allowed values:
- instrument: acoustic_grand_piano, piano, acoustic_guitar_nylon, guitar, violin, strings, flute, drums
- mood: neutral, happy, sad, calm, energetic, dark
- complexity: simple, moderate, complex
- polyphony_level: very_low, low, medium, high
- density_level: very_low, low, medium, high, very_high
- register: low, mid, mid_high, high
- note_duration_level: short, medium, long
- genre: any, ambient, blues, classical, country, folk, hip_hop, house, jazz, latin, pop, reggae, rock, techno, trance, world

Interpretation guidelines:
- simple melody -> complexity=simple, polyphony_level=very_low or low, note_duration_level=medium or long
- low polyphony / monophonic -> polyphony_level=very_low or low
- dense / busy -> density_level=high or very_high
- sparse / minimal -> density_level=very_low or low
- happy -> mood=happy
- energetic -> mood=energetic
- calm -> mood=calm
- sad -> mood=sad
- dark -> mood=dark
- piano -> acoustic_grand_piano or piano
- guitar -> acoustic_guitar_nylon or guitar
- violin -> violin
- strings -> strings
- flute -> flute
- drums -> drums

Defaults when uncertain:
- instrument: acoustic_grand_piano
- mood: neutral
- complexity: moderate
- polyphony_level: medium
- density_level: medium
- register: mid
- note_duration_level: medium
- genre: any
""".strip()


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "instrument": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["instrument"])},
        "mood": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["mood"])},
        "complexity": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["complexity"])},
        "polyphony_level": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["polyphony_level"])},
        "density_level": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["density_level"])},
        "register": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["register"])},
        "note_duration_level": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["note_duration_level"])},
        "genre": {"type": "string", "enum": sorted(ALLOWED_VARIABLES["genre"])},
    },
    "required": [
        "instrument",
        "mood",
        "complexity",
        "polyphony_level",
        "density_level",
        "register",
        "note_duration_level",
        "genre",
    ]
}


def generate_json_from_prompt(prompt: str, model_name: str) -> Dict[str, Any]:
    # request payload to send to the LLM API
    payload = {
        "model": model_name,
        "system": build_system_prompt(),
        "prompt": prompt,
        "format": LLM_SCHEMA,
        "stream": False,
    }

    # Send a POST request
    resp = requests.post("http://localhost:11434/api/generate", json=payload)
    print("OLLAMA STATUS:", resp.status_code)

    # Raise an exception if the request failed
    resp.raise_for_status()

    # Parse the JSON response
    data = resp.json()

    # Extract and clean the raw text output
    raw = data["response"].strip()

    try:
        # Parse the response as valid JSON
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)


def main() -> None:
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_input", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="happy piano, simple melody, low polyphony",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen2.5:14b-instruct",
        help="Choose the ollama model to use"
    )
    parser.add_argument(
        "--generation_mode",
        choices=["tail_infill", "full_conditional", "autoregressive"],
        default="tail_infill",
        help=("Generation strategy used by MIDIGPT:\n\n"
              "tail_infill:\n"
              "  Uses the first bar as context and generates the remaining bars.\n"
              "full_conditional:\n"
              "  Generates all bars from scratch, conditioned only by the prompt.\n"
              "autoregressive:\n"
              "  Generates all bars sequentially (step-by-step generation).\n"
              ),
    )
    args = parser.parse_args()

    # Get variables
    midi_input = args.midi_input
    prompt = args.prompt
    ckpt = args.ckpt
    midi_dest = args.output_dir

    # Convert MIDI to Json file
    encoder = midigpt.ExpressiveEncoder()
    midi_json_input = json.loads(encoder.midi_to_json(midi_input))

    print("\n=== USER PROMPT ===")
    print(prompt)

    # Generate raw json from user prompt
    print("\n=== RAW LLM JSON ===")
    llm_json_raw = generate_json_from_prompt(prompt, args.ollama_model)
    print(json.dumps(llm_json_raw, indent=2))

    # Clean prompt
    print("\n=== SANITIZED JSON ===")
    semantic_controls = sanitize_raw_json(llm_json_raw)
    print(json.dumps(semantic_controls, indent=2))

    # Build parameters for MIDI-GPT
    print("\n=== VALID STATUS ===")
    valid_status = build_valid_status(midi_json_input, semantic_controls, args.generation_mode)
    print(json.dumps(valid_status, indent=2))

    print("\n=== PARAMI ===")
    parami = build_parami(ckpt, args.generation_mode)
    print(json.dumps(parami, indent=2))

    # Serialize input to JSON strings
    piece = json.dumps(midi_json_input)
    status = json.dumps(valid_status)
    param = json.dumps(parami)

    # Initialize callback manager
    callbacks = midigpt.CallbackManager()
    max_attempts = 3

    # Generate MIDI data
    midi_str = midigpt.sample_multi_step(piece, status, param, max_attempts, callbacks)
    midi_str = midi_str[0]
    _ = json.loads(midi_str)

    # Convert JSON to MIDI
    encoder = midigpt.ExpressiveEncoder()
    encoder.json_to_midi(midi_str, midi_dest)

    print(f"\nGenerated MIDI saved to: {midi_dest}")


if __name__ == "__main__":
    main()
