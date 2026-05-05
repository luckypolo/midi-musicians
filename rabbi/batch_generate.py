"""
Batch MIDI Generation for Three-Condition Evaluation
=====================================================
Generates MIDI files for each prompt in prompts.json under two conditions:
  Condition 1 (direct_controls): Uses the ground-truth 'controls' field
      directly to build MIDI-GPT status/params — no LLM involved.
  Condition 2 (prompt_to_dsl): Runs parse_prompt() on the 'text' field,
      producing DSL controls heuristically, then builds MIDI-GPT status/params.

Each prompt is generated with multiple random seeds for variance estimation.

Setup:
  1. Clone your midi-musicians repo, switch to the mike branch
  2. Copy this script + prompts.json into the repo root
  3. Make sure midigpt library is installed (Python 3.8 env)
  4. Make sure models/model.ckpt exists (extract from models/model.zip)
  5. Place a seed MIDI file (e.g., seed_input.mid) in the repo root —
     this is the MIDI file MIDI-GPT uses as context for generation

Usage:
  python batch_generate.py --input_midi seed_input.mid --prompts prompts.json --seeds 3
  python batch_generate.py --input_midi seed_input.mid --prompts prompts.json --seeds 3 --condition 1
  python batch_generate.py --input_midi seed_input.mid --prompts prompts.json --seeds 3 --condition 2

Output structure:
  outputs/
    condition1_direct_controls/
      p01_seed1.mid
      p01_seed2.mid
      ...
    condition2_prompt_to_dsl/
      p01_seed1.mid
      ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import copy
from pathlib import Path

# ============================================================
# MIDI-GPT imports — adjust if your repo structure differs
# ============================================================
try:
    import midigpt
except ImportError:
    print("ERROR: midigpt library not found.")
    print("Make sure you're in the Python 3.8 environment where midigpt is installed.")
    print("Run: bash midigpt_setup_helper.sh -i -c -d midigpt_dir")
    sys.exit(1)

from prompt_to_controls import parse_prompt
from midigpt_bridge import (
    build_status,
    default_params,
    INSTRUMENT_TO_MIDIGPT,
    DENSITY_TO_SCORE,
    POLYPHONY_TO_LIMIT,
    temperature_from_predictions,
)


# ============================================================
# Configuration
# ============================================================
DEFAULT_CKPT = "models/model.ckpt"
OUTPUT_ROOT = "outputs"
SEEDS = [42, 123, 456, 789, 1024]  # pool of seeds to draw from


def load_piece(midi_path: str) -> str:
    """Load a MIDI file into MIDI-GPT's JSON piece representation."""
    return midigpt.midi_to_piece(midi_path)


def generate_one(
    piece_json: str,
    predictions: dict,
    ckpt: str,
    seed: int,
    output_path: str,
    generation_mode: str = "tail_infill",
):
    """
    Run MIDI-GPT generation with given controls and save the output.

    Args:
        piece_json: JSON string of the loaded MIDI piece
        predictions: Dict with keys matching DSL variables
            (instrument, mood, density_level, polyphony_level, etc.)
        ckpt: Path to model checkpoint
        seed: Random seed for reproducibility
        output_path: Where to save the generated MIDI file
        generation_mode: One of 'tail_infill', 'full_conditional', 'autoregressive'
    """
    # Build status from predictions
    # Determine which bars to select based on generation mode
    if generation_mode == "tail_infill":
        # Keep first bar as context, generate the rest
        selected_bars = [False, True, True, True]
    elif generation_mode == "full_conditional":
        selected_bars = [True, True, True, True]
    else:
        selected_bars = [False, True, True, True]

    status = build_status(predictions, track_id=0, selected_bars=selected_bars)

    # Build params
    params = default_params(ckpt)
    params["sampling_seed"] = seed

    # If autoregressive mode, set the flag
    if generation_mode == "autoregressive":
        status["tracks"][0]["autoregressive"] = True

    # Run generation
    try:
        piece = json.loads(piece_json)
        result = midigpt.sample(
            json.dumps(piece),
            json.dumps(status),
            json.dumps(params),
        )

        # Save output MIDI
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        midigpt.piece_to_midi(result, output_path)
        return True

    except Exception as e:
        print(f"    ERROR generating {output_path}: {e}")
        return False


def controls_to_predictions(controls: dict) -> dict:
    """
    Convert a controls dict from prompts.json into the predictions
    format expected by build_status().

    The controls dict has all 8 DSL fields. This function ensures
    they're in the right format for the bridge.
    """
    return {
        "instrument": controls.get("instrument", "piano"),
        "mood": controls.get("mood", "neutral"),
        "genre": controls.get("genre", "unknown"),
        "density_level": controls.get("density_level", "medium"),
        "polyphony_level": controls.get("polyphony_level", "medium"),
        "note_duration_level": controls.get("note_duration_level", "medium"),
        "register": controls.get("register", "medium"),
        "complexity": controls.get("complexity", "medium"),
    }


def parse_prompt_to_predictions(text: str) -> dict:
    """
    Run parse_prompt() on a text string and convert the result
    into the predictions format expected by build_status().
    """
    pc = parse_prompt(text)
    return {
        "instrument": pc.instrument,
        "mood": pc.mood,
        "genre": pc.genre if pc.genre != "unknown" else "other",
        "density_level": pc.density_level,
        "polyphony_level": pc.polyphony_level,
        "note_duration_level": pc.note_duration_level,
        "register": pc.register,
        "complexity": pc.complexity,
    }


def run_condition(
    prompts: list,
    condition_name: str,
    condition_num: int,
    piece_json: str,
    ckpt: str,
    num_seeds: int,
    generation_mode: str,
):
    """Run generation for all prompts under one condition."""
    output_dir = os.path.join(OUTPUT_ROOT, f"condition{condition_num}_{condition_name}")
    os.makedirs(output_dir, exist_ok=True)

    total = len(prompts) * num_seeds
    completed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"CONDITION {condition_num}: {condition_name}")
    print(f"{'='*60}")
    print(f"Prompts: {len(prompts)}, Seeds per prompt: {num_seeds}")
    print(f"Total generations: {total}")
    print(f"Output directory: {output_dir}")
    print(f"Generation mode: {generation_mode}")
    print()

    # Also save the controls used for each prompt (for analysis)
    controls_log = []

    for prompt in prompts:
        prompt_id = prompt["id"]
        text = prompt["text"]

        # Get predictions based on condition
        if condition_num == 1:
            predictions = controls_to_predictions(prompt["controls"])
            source = "direct_controls"
        else:
            predictions = parse_prompt_to_predictions(text)
            source = "parse_prompt_heuristic"

        controls_log.append({
            "id": prompt_id,
            "text": text,
            "source": source,
            "predictions": predictions,
        })

        print(f"  [{prompt_id}] {text[:50]}...")
        print(f"    Controls: mood={predictions['mood']}, "
              f"density={predictions['density_level']}, "
              f"instrument={predictions['instrument']}, "
              f"register={predictions['register']}")

        for seed_idx in range(num_seeds):
            seed = SEEDS[seed_idx % len(SEEDS)]
            output_path = os.path.join(output_dir, f"{prompt_id}_seed{seed_idx + 1}.mid")

            print(f"    Generating seed {seed_idx + 1}/{num_seeds} "
                  f"(seed={seed}) -> {os.path.basename(output_path)}", end=" ")

            success = generate_one(
                piece_json=piece_json,
                predictions=predictions,
                ckpt=ckpt,
                seed=seed,
                output_path=output_path,
                generation_mode=generation_mode,
            )

            if success:
                completed += 1
                print("[OK]")
            else:
                failed += 1
                print("[FAILED]")

    # Save controls log
    log_path = os.path.join(output_dir, "_controls_log.json")
    with open(log_path, "w") as f:
        json.dump(controls_log, f, indent=2)

    print(f"\nCondition {condition_num} complete: {completed}/{total} succeeded, {failed} failed")
    print(f"Controls log saved to {log_path}")
    return completed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Batch MIDI generation for three-condition evaluation"
    )
    parser.add_argument(
        "--input_midi", required=True,
        help="Seed MIDI file used as context for MIDI-GPT generation"
    )
    parser.add_argument(
        "--prompts", default="prompts.json",
        help="Path to prompts.json file"
    )
    parser.add_argument(
        "--ckpt", default=DEFAULT_CKPT,
        help="Path to MIDI-GPT model checkpoint"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds per prompt (default: 3)"
    )
    parser.add_argument(
        "--condition", type=int, choices=[1, 2], default=None,
        help="Run only one condition (1=direct, 2=prompt_to_dsl). "
             "Default: run both."
    )
    parser.add_argument(
        "--mode", default="tail_infill",
        choices=["tail_infill", "full_conditional", "autoregressive"],
        help="MIDI-GPT generation mode (default: tail_infill)"
    )
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Load the seed MIDI piece
    print(f"Loading seed MIDI: {args.input_midi}")
    piece_json = load_piece(args.input_midi)
    print("Seed MIDI loaded successfully")

    # Run conditions
    conditions_to_run = []
    if args.condition is None or args.condition == 1:
        conditions_to_run.append((1, "direct_controls"))
    if args.condition is None or args.condition == 2:
        conditions_to_run.append((2, "prompt_to_dsl"))

    total_completed = 0
    total_failed = 0

    for cond_num, cond_name in conditions_to_run:
        completed, failed = run_condition(
            prompts=prompts,
            condition_name=cond_name,
            condition_num=cond_num,
            piece_json=piece_json,
            ckpt=args.ckpt,
            num_seeds=args.seeds,
            generation_mode=args.mode,
        )
        total_completed += completed
        total_failed += failed

    # Summary
    print(f"\n{'='*60}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total generated: {total_completed}")
    print(f"Total failed:    {total_failed}")
    print(f"Output directory: {OUTPUT_ROOT}/")
    print()
    print("Next steps:")
    print("  1. Copy the outputs/ folder to your clamp3/ directory")
    print("  2. Run CLaMP 3 evaluation:")
    print(f"     python eval_pipeline.py --compare \\")
    print(f"       --direct_controls_dir outputs/condition1_direct_controls \\")
    print(f"       --prompt_to_dsl_dir outputs/condition2_prompt_to_dsl \\")
    print(f"       --prompts_file {args.prompts}")


if __name__ == "__main__":
    main()
