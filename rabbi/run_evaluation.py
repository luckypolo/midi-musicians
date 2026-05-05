"""
Prepare and Evaluate Two-Condition MIDI Generation Experiment
==============================================================
Run this from your clamp3/ directory.

It expects:
  - outputs/ folder containing files like p01_controls_seed1.mid, p01_prompt_seed1.mid, etc.
    (place it inside midicaps_subset/midi/ or directly in clamp3/)
  - prompts.json in clamp3/

Usage:
  python run_evaluation.py --outputs_dir midicaps_subset/midi/outputs --prompts prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


CLAMP3_ROOT = os.path.dirname(os.path.abspath(__file__))


def cosine_similarity(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_clamp3_embed(input_dir, output_dir, label=""):
    """Run CLaMP 3 embedding extraction."""
    os.makedirs(output_dir, exist_ok=True)
    code_dir = os.path.join(CLAMP3_ROOT, "code")
    cmd = [sys.executable, "extract_clamp3.py",
           os.path.abspath(input_dir), os.path.abspath(output_dir), "--get_global"]
    print(f"  [{label}] Extracting embeddings...")
    subprocess.run(cmd, cwd=code_dir, check=True)


def run_midi_to_mtf(input_dir, output_dir):
    """Convert MIDI to MTF format."""
    os.makedirs(output_dir, exist_ok=True)
    preprocessing_dir = os.path.join(CLAMP3_ROOT, "preprocessing", "midi")
    cmd = [sys.executable, "batch_midi2mtf.py",
           os.path.abspath(input_dir), os.path.abspath(output_dir), "--m3_compatible"]
    print(f"  Converting MIDI to MTF...")
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", required=True,
                        help="Directory with MIDI files named like p01_controls_seed1.mid")
    parser.add_argument("--prompts", default="prompts.json")
    args = parser.parse_args()

    outputs_dir = os.path.abspath(args.outputs_dir)
    eval_dir = os.path.join(CLAMP3_ROOT, "eval_experiment")

    # Load prompts
    with open(args.prompts) as f:
        all_prompts = json.load(f)
    prompts_by_id = {p["id"]: p for p in all_prompts}

    # Discover which prompts we have
    midi_files = list(Path(outputs_dir).glob("*.mid"))
    prompt_ids = sorted(set(f.stem.split("_")[0] for f in midi_files))
    print(f"Found {len(midi_files)} MIDI files for prompts: {prompt_ids}")

    # ---- Step 1: Split into condition folders ----
    cond1_dir = os.path.join(eval_dir, "condition1_midi")
    cond2_dir = os.path.join(eval_dir, "condition2_midi")
    texts_dir = os.path.join(eval_dir, "prompt_texts")

    for d in [cond1_dir, cond2_dir, texts_dir]:
        os.makedirs(d, exist_ok=True)

    for f in midi_files:
        name = f.stem
        if "_controls_" in name:
            shutil.copy(f, os.path.join(cond1_dir, f.name))
        elif "_prompt_" in name:
            shutil.copy(f, os.path.join(cond2_dir, f.name))

    print(f"  Condition 1 (direct controls): {len(list(Path(cond1_dir).glob('*.mid')))} files")
    print(f"  Condition 2 (prompt to DSL):   {len(list(Path(cond2_dir).glob('*.mid')))} files")

    # ---- Step 2: Create text prompt files for matching ----
    for pid in prompt_ids:
        if pid in prompts_by_id:
            txt_path = os.path.join(texts_dir, f"{pid}.txt")
            with open(txt_path, "w") as f:
                f.write(prompts_by_id[pid]["text"])

    print(f"  Created {len(prompt_ids)} text prompt files")

    # ---- Step 3: Convert MIDI to MTF ----
    cond1_mtf = os.path.join(eval_dir, "condition1_mtf")
    cond2_mtf = os.path.join(eval_dir, "condition2_mtf")
    run_midi_to_mtf(cond1_dir, cond1_mtf)
    run_midi_to_mtf(cond2_dir, cond2_mtf)

    # ---- Step 4: Extract CLaMP 3 embeddings ----
    cond1_emb = os.path.join(eval_dir, "condition1_embeddings")
    cond2_emb = os.path.join(eval_dir, "condition2_embeddings")
    text_emb = os.path.join(eval_dir, "text_embeddings")

    run_clamp3_embed(cond1_mtf, cond1_emb, "cond1")
    run_clamp3_embed(cond2_mtf, cond2_emb, "cond2")
    run_clamp3_embed(texts_dir, text_emb, "texts")

    # ---- Step 5: Compute similarities ----
    print("\nComputing similarity matrices...")

    # Load all embeddings
    def load_embeddings(emb_dir, prefix_strip=""):
        embs = {}
        for f in sorted(Path(emb_dir).glob("*.npy")):
            name = f.stem
            if prefix_strip and name.startswith(prefix_strip):
                name = name[len(prefix_strip):]
            embs[name] = np.load(f)
        return embs

    cond1_embeddings = load_embeddings(cond1_emb, "mid-")
    cond2_embeddings = load_embeddings(cond2_emb, "mid-")
    text_embeddings = load_embeddings(text_emb, "txt-")

    # ---- Table 1: Text alignment per prompt per condition ----
    # For each prompt, average the cosine similarity of its 3 seeds against its text
    results = []
    for pid in prompt_ids:
        if pid not in prompts_by_id or pid not in text_embeddings:
            continue

        text_emb_vec = text_embeddings[pid]
        prompt_text = prompts_by_id[pid]["text"]

        # Condition 1 seeds
        c1_sims = []
        for key, emb in cond1_embeddings.items():
            if key.startswith(pid + "_controls"):
                c1_sims.append(cosine_similarity(emb, text_emb_vec))

        # Condition 2 seeds
        c2_sims = []
        for key, emb in cond2_embeddings.items():
            if key.startswith(pid + "_prompt"):
                c2_sims.append(cosine_similarity(emb, text_emb_vec))

        results.append({
            "prompt_id": pid,
            "text": prompt_text[:60],
            "cond1_mean": np.mean(c1_sims) if c1_sims else 0,
            "cond1_std": np.std(c1_sims) if c1_sims else 0,
            "cond2_mean": np.mean(c2_sims) if c2_sims else 0,
            "cond2_std": np.std(c2_sims) if c2_sims else 0,
            "cond1_seeds": c1_sims,
            "cond2_seeds": c2_sims,
        })

    # ---- Table 2: Cross-condition MIDI-to-MIDI similarity ----
    midi_midi_results = []
    for pid in prompt_ids:
        for seed in ["seed1", "seed2", "seed3"]:
            c1_key = f"{pid}_controls_{seed}"
            c2_key = f"{pid}_prompt_{seed}"
            if c1_key in cond1_embeddings and c2_key in cond2_embeddings:
                sim = cosine_similarity(cond1_embeddings[c1_key], cond2_embeddings[c2_key])
                midi_midi_results.append({"prompt_id": pid, "seed": seed, "similarity": sim})

    # ---- Print results ----
    print(f"\n{'='*90}")
    print("TEXT ALIGNMENT: Cosine similarity between prompt text and generated MIDI")
    print(f"{'='*90}")
    print(f"{'Prompt':<8} {'Text':<45} {'Cond1 (direct)':<18} {'Cond2 (prompt)':<18} {'Delta':<10}")
    print("-" * 90)

    c1_all = []
    c2_all = []
    for r in results:
        delta = r["cond2_mean"] - r["cond1_mean"]
        c1_all.append(r["cond1_mean"])
        c2_all.append(r["cond2_mean"])
        print(f"{r['prompt_id']:<8} {r['text']:<45} "
              f"{r['cond1_mean']:.4f} ±{r['cond1_std']:.3f}  "
              f"{r['cond2_mean']:.4f} ±{r['cond2_std']:.3f}  "
              f"{delta:+.4f}")

    print("-" * 90)
    print(f"{'AVERAGE':<8} {'':<45} "
          f"{np.mean(c1_all):.4f}             "
          f"{np.mean(c2_all):.4f}             "
          f"{np.mean(c2_all) - np.mean(c1_all):+.4f}")

    print(f"\n{'='*70}")
    print("CROSS-CONDITION: MIDI-to-MIDI similarity (same prompt, same seed)")
    print(f"{'='*70}")
    print(f"{'Prompt':<8} {'Seed':<8} {'Similarity':<12}")
    print("-" * 30)
    for r in midi_midi_results:
        print(f"{r['prompt_id']:<8} {r['seed']:<8} {r['similarity']:.4f}")

    if midi_midi_results:
        avg_mm = np.mean([r["similarity"] for r in midi_midi_results])
        print(f"\nAverage MIDI-MIDI similarity: {avg_mm:.4f}")
        print("(Higher = conditions produce similar output; lower = the prompting layer changes generation)")

    # ---- Step 6: Log to wandb ----
    try:
        import wandb

        run = wandb.init(
            project="midi-nlp-evaluation",
            name="two_condition_comparison",
            config={
                "model": "CLaMP 3 (C2)",
                "prompts_evaluated": prompt_ids,
                "seeds_per_prompt": 3,
                "conditions": ["direct_controls", "prompt_to_dsl"],
            }
        )

        # Text alignment table
        columns = ["prompt_id", "text", "cond1_mean", "cond1_std",
                    "cond2_mean", "cond2_std", "delta"]
        table = wandb.Table(columns=columns)
        for r in results:
            delta = r["cond2_mean"] - r["cond1_mean"]
            table.add_data(r["prompt_id"], r["text"],
                           round(r["cond1_mean"], 4), round(r["cond1_std"], 4),
                           round(r["cond2_mean"], 4), round(r["cond2_std"], 4),
                           round(delta, 4))
        wandb.log({"text_alignment_comparison": table})

        # MIDI-MIDI table
        mm_table = wandb.Table(columns=["prompt_id", "seed", "midi_midi_similarity"])
        for r in midi_midi_results:
            mm_table.add_data(r["prompt_id"], r["seed"], round(r["similarity"], 4))
        wandb.log({"cross_condition_similarity": mm_table})

        # Summary metrics
        wandb.log({
            "cond1_avg_text_alignment": round(float(np.mean(c1_all)), 4),
            "cond2_avg_text_alignment": round(float(np.mean(c2_all)), 4),
            "delta_text_alignment": round(float(np.mean(c2_all) - np.mean(c1_all)), 4),
            "avg_midi_midi_similarity": round(float(avg_mm), 4) if midi_midi_results else 0,
        })

        # Bar chart: condition comparison per prompt
        bar_data = [[r["prompt_id"], r["cond1_mean"], r["cond2_mean"]] for r in results]
        bar_table = wandb.Table(data=bar_data,
                                columns=["prompt", "direct_controls", "prompt_to_dsl"])
        wandb.log({
            "condition_comparison": wandb.plot.bar(
                bar_table, "prompt", "direct_controls",
                title="Text alignment: direct controls vs prompt-to-DSL"
            )
        })

        # Heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            for ax, (cond_name, cond_embs, suffix) in zip(axes, [
                ("Condition 1: Direct controls", cond1_embeddings, "controls"),
                ("Condition 2: Prompt to DSL", cond2_embeddings, "prompt"),
            ]):
                # Build matrix: rows = midi files for this condition, cols = text prompts
                midi_keys = sorted([k for k in cond_embs.keys() if suffix in k])
                text_keys = sorted(text_embeddings.keys())

                matrix = np.zeros((len(midi_keys), len(text_keys)))
                for i, mk in enumerate(midi_keys):
                    for j, tk in enumerate(text_keys):
                        matrix[i, j] = cosine_similarity(cond_embs[mk], text_embeddings[tk])

                im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.35)
                ax.set_xticks(range(len(text_keys)))
                ax.set_xticklabels(text_keys, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(midi_keys)))
                ax.set_yticklabels([k.replace(f"_{suffix}", "") for k in midi_keys], fontsize=7)
                ax.set_title(cond_name, fontsize=10)

                for i in range(len(midi_keys)):
                    for j in range(len(text_keys)):
                        ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                                fontsize=5, color="white" if matrix[i,j] > 0.18 else "black")

            plt.colorbar(im, ax=axes, label="Cosine similarity", shrink=0.8)
            plt.suptitle("CLaMP 3 Text Alignment: Two-Condition Comparison", fontsize=12)
            plt.tight_layout()

            heatmap_path = os.path.join(eval_dir, "comparison_heatmap.png")
            fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            wandb.log({"comparison_heatmap": wandb.Image(heatmap_path)})
            print(f"\nSaved heatmap to {heatmap_path}")
        except ImportError:
            print("matplotlib not installed, skipping heatmap")

        wandb.finish()
        print(f"Wandb run complete! View at: {run.url}")

    except ImportError:
        print("\nwandb not installed — results printed above only.")

    # Save results JSON
    results_path = os.path.join(eval_dir, "evaluation_results.json")
    save_data = {
        "text_alignment": results,
        "midi_midi_similarity": midi_midi_results,
        "summary": {
            "cond1_avg": float(np.mean(c1_all)),
            "cond2_avg": float(np.mean(c2_all)),
            "delta": float(np.mean(c2_all) - np.mean(c1_all)),
            "avg_midi_midi": float(avg_mm) if midi_midi_results else 0,
        }
    }
    # Convert numpy to float for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=convert)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
