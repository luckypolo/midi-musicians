"""
CLaMP 3 Evaluation Pipeline with Wandb Logging
================================================
This script:
1. Downloads a subset of MIDICaps (captions + MIDI files)
2. Embeds your generated MIDI files + MIDICaps references with CLaMP 3
3. Computes cosine similarity between prompts and generated MIDI
4. Logs similarity matrices, tables, and charts to wandb

Prerequisites:
- CLaMP 3 repo cloned and C2 weights in code/ folder
- config.py line 66 set to "c2"
- conda environment with clamp3 deps installed
- pip install wandb faiss-cpu

Usage:
  Place this script in your clamp3/ repo root.
  
  # Step 1: Download MIDICaps subset
  python eval_pipeline.py --download --subset_size 200

  # Step 2: Run evaluation on your generated MIDI files
  python eval_pipeline.py --evaluate --generated_dir path/to/your/generated/midi
  
  # Step 3: Run the full comparison (3 conditions)
  python eval_pipeline.py --compare \
      --direct_controls_dir path/to/condition1 \
      --prompt_to_dsl_dir path/to/condition2 \
      --rag_augmented_dir path/to/condition3 \
      --prompts_file prompts.json
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np

CLAMP3_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CLAMP3_ROOT, "cache")
MIDICAPS_DIR = os.path.join(CLAMP3_ROOT, "midicaps_subset")
MIDICAPS_MIDI_DIR = os.path.join(MIDICAPS_DIR, "midi")
MIDICAPS_MTF_DIR = os.path.join(MIDICAPS_DIR, "mtf")
MIDICAPS_TEXTS_DIR = os.path.join(MIDICAPS_DIR, "texts")
MIDICAPS_EMBEDDINGS_DIR = os.path.join(MIDICAPS_DIR, "embeddings")
RESULTS_DIR = os.path.join(CLAMP3_ROOT, "eval_results")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_embedding(npy_path):
    """Load a .npy embedding file."""
    return np.load(npy_path)


def download_midicaps_subset(subset_size=200):
    """
    Downloads train.json from MIDICaps, selects a diverse subset,
    and downloads only those MIDI files.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_xet]"])
        from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(MIDICAPS_MIDI_DIR, exist_ok=True)
    os.makedirs(MIDICAPS_TEXTS_DIR, exist_ok=True)

    # Download the captions JSON
    print("Downloading MIDICaps train.json (captions)...")
    json_path = hf_hub_download(
        repo_id="amaai-lab/MidiCaps",
        filename="train.json",
        repo_type="dataset",
        local_dir=MIDICAPS_DIR
    )

    # Load and parse captions
    # train.json can be: single JSON object, JSON array, or JSONL (one JSON per line)
    print("Parsing captions...")
    entries = []
    with open(json_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "{":
            # Try single JSON object first, fall back to JSONL
            try:
                data = json.load(f)
                entries = [(k, v) for k, v in data.items()]
            except json.JSONDecodeError:
                # JSONL format: one JSON object per line
                f.seek(0)
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        filename = item.get("location", item.get("filename", f"track_{i}"))
                        entries.append((filename, item))
                    except json.JSONDecodeError:
                        continue
        elif first_char == "[":
            data = json.load(f)
            entries = [(item.get("location", item.get("filename", f"track_{i}")), item)
                       for i, item in enumerate(data)]
        else:
            # JSONL format
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    filename = item.get("location", item.get("filename", f"track_{i}"))
                    entries.append((filename, item))
                except json.JSONDecodeError:
                    continue

    if not entries:
        print("Error: Could not parse any entries from train.json")
        return

    print(f"Total entries in MIDICaps: {len(entries)}")

    # Select a diverse subset based on mood/genre keywords
    mood_keywords = {
        "happy": [], "sad": [], "energetic": [], "calm": [],
        "dark": [], "aggressive": [], "melodic": [], "ambient": [],
        "romantic": [], "epic": []
    }

    for filename, meta in entries:
        caption = meta.get("caption", "") if isinstance(meta, dict) else str(meta)
        caption_lower = caption.lower()
        for mood in mood_keywords:
            if mood in caption_lower and len(mood_keywords[mood]) < subset_size // len(mood_keywords):
                mood_keywords[mood].append((filename, meta))
                break

    # Collect selected entries
    selected = []
    for mood, items in mood_keywords.items():
        selected.extend(items)
        print(f"  {mood}: {len(items)} files")

    # Fill remaining with random entries if needed
    if len(selected) < subset_size:
        import random
        random.seed(42)
        remaining = [e for e in entries if e not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:subset_size - len(selected)])

    selected = selected[:subset_size]
    print(f"\nSelected {len(selected)} files for subset")

    # Save captions as individual text files and a combined JSON
    subset_meta = {}
    for filename, meta in selected:
        caption = meta.get("caption", "") if isinstance(meta, dict) else str(meta)

        # Clean filename for text file
        safe_name = Path(filename).stem
        txt_path = os.path.join(MIDICAPS_TEXTS_DIR, f"{safe_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        subset_meta[filename] = {
            "caption": caption,
            "safe_name": safe_name
        }
        if isinstance(meta, dict):
            for key in ["genre", "mood", "tempo", "key", "time_signature"]:
                if key in meta:
                    subset_meta[filename][key] = meta[key]

    # Save subset metadata
    meta_path = os.path.join(MIDICAPS_DIR, "subset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(subset_meta, f, indent=2)
    print(f"Saved subset metadata to {meta_path}")

    # Download the tar.gz and extract only the selected files
    print("\nDownloading midicaps.tar.gz (1.62 GB)... this may take a while.")
    tar_path = hf_hub_download(
        repo_id="amaai-lab/MidiCaps",
        filename="midicaps.tar.gz",
        repo_type="dataset",
        local_dir=MIDICAPS_DIR
    )

    print("Extracting selected MIDI files...")
    selected_filenames = {fn for fn, _ in selected}
    extracted_count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            member_name = member.name
            # Check if this file is in our selected set
            base = os.path.basename(member_name)
            matches = [fn for fn in selected_filenames
                       if base in fn or fn in member_name]
            if matches or base.endswith((".mid", ".midi")):
                if extracted_count < subset_size:
                    # Extract to midi dir
                    member.name = os.path.basename(member.name)
                    tar.extract(member, MIDICAPS_MIDI_DIR)
                    extracted_count += 1

                    if extracted_count % 50 == 0:
                        print(f"  Extracted {extracted_count} files...")

    print(f"Extracted {extracted_count} MIDI files to {MIDICAPS_MIDI_DIR}")
    print("\nMIDICaps subset ready! Next steps:")
    print("  python eval_pipeline.py --evaluate --generated_dir <your_midi_dir>")


def convert_and_embed(midi_dir, mtf_dir, embed_dir, label=""):
    """Convert MIDI files to MTF, then extract CLaMP 3 embeddings."""
    os.makedirs(mtf_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    # Convert MIDI to MTF
    print(f"\n[{label}] Converting MIDI to MTF...")
    preprocessing_dir = os.path.join(CLAMP3_ROOT, "preprocessing", "midi")
    cmd = [
        sys.executable, "batch_midi2mtf.py",
        os.path.abspath(midi_dir),
        os.path.abspath(mtf_dir),
        "--m3_compatible"
    ]
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)

    # Count MTF files
    mtf_files = list(Path(mtf_dir).glob("*.mtf"))
    print(f"  Created {len(mtf_files)} MTF files")

    # Extract embeddings
    print(f"[{label}] Extracting CLaMP 3 embeddings...")
    code_dir = os.path.join(CLAMP3_ROOT, "code")
    cmd = [
        sys.executable, "extract_clamp3.py",
        os.path.abspath(mtf_dir),
        os.path.abspath(embed_dir),
        "--get_global"
    ]
    subprocess.run(cmd, cwd=code_dir, check=True)

    # Count embeddings
    npy_files = list(Path(embed_dir).glob("*.npy"))
    print(f"  Created {len(npy_files)} embedding files")
    return npy_files


def embed_texts(texts_dir, embed_dir, label=""):
    """Extract CLaMP 3 embeddings for text files."""
    os.makedirs(embed_dir, exist_ok=True)

    print(f"\n[{label}] Extracting text embeddings...")
    code_dir = os.path.join(CLAMP3_ROOT, "code")
    cmd = [
        sys.executable, "extract_clamp3.py",
        os.path.abspath(texts_dir),
        os.path.abspath(embed_dir),
        "--get_global"
    ]
    subprocess.run(cmd, cwd=code_dir, check=True)

    npy_files = list(Path(embed_dir).glob("*.npy"))
    print(f"  Created {len(npy_files)} text embedding files")
    return npy_files


def compute_similarity_matrix(midi_embed_dir, text_embed_dir):
    """
    Compute cosine similarity between all MIDI embeddings
    and all text embeddings.

    Returns:
        midi_names: list of MIDI file stems
        text_names: list of text file stems
        matrix: np.array of shape (n_midi, n_text)
    """
    midi_files = sorted(Path(midi_embed_dir).glob("*.npy"))
    text_files = sorted(Path(text_embed_dir).glob("*.npy"))

    midi_names = [f.stem.replace("mid-", "") for f in midi_files]
    text_names = [f.stem.replace("txt-", "") for f in text_files]

    matrix = np.zeros((len(midi_files), len(text_files)))

    for i, mf in enumerate(midi_files):
        midi_emb = load_embedding(mf)
        for j, tf in enumerate(text_files):
            text_emb = load_embedding(tf)
            matrix[i, j] = cosine_similarity(midi_emb, text_emb)

    return midi_names, text_names, matrix


def compute_midi_midi_similarity(embed_dir_a, embed_dir_b):
    """
    Compute pairwise MIDI-to-MIDI cosine similarity
    between two sets of generated outputs.
    Matches files by name (same prompt, same seed).
    """
    files_a = {f.stem: f for f in Path(embed_dir_a).glob("*.npy")}
    files_b = {f.stem: f for f in Path(embed_dir_b).glob("*.npy")}

    common = sorted(set(files_a.keys()) & set(files_b.keys()))
    if not common:
        # Try matching by removing prefix
        strip_a = {f.stem.split("-", 1)[-1]: f for f in Path(embed_dir_a).glob("*.npy")}
        strip_b = {f.stem.split("-", 1)[-1]: f for f in Path(embed_dir_b).glob("*.npy")}
        common = sorted(set(strip_a.keys()) & set(strip_b.keys()))
        files_a, files_b = strip_a, strip_b

    results = {}
    for name in common:
        emb_a = load_embedding(files_a[name])
        emb_b = load_embedding(files_b[name])
        results[name] = cosine_similarity(emb_a, emb_b)

    return results


def log_to_wandb(midi_names, text_names, sim_matrix,
                 condition_label="", extra_metrics=None,
                 midi_midi_sims=None):
    """Log evaluation results to wandb."""
    try:
        import wandb
    except ImportError:
        print("Installing wandb...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb

    # Initialize wandb run
    run = wandb.init(
        project="midi-nlp-evaluation",
        name=f"clamp3_eval_{condition_label}" if condition_label else "clamp3_eval",
        config={
            "model": "CLaMP 3 (C2)",
            "condition": condition_label,
            "n_midi_files": len(midi_names),
            "n_text_prompts": len(text_names),
        }
    )

    # --- Table 1: Full similarity matrix ---
    columns = ["midi_file"] + text_names
    table = wandb.Table(columns=columns)
    for i, midi_name in enumerate(midi_names):
        row = [midi_name] + [round(float(sim_matrix[i, j]), 4)
                              for j in range(len(text_names))]
        table.add_data(*row)
    wandb.log({"similarity_matrix": table})

    # --- Table 2: Per-file best match ---
    best_match_columns = ["midi_file", "best_text_match", "similarity", "rank"]
    best_match_table = wandb.Table(columns=best_match_columns)
    for i, midi_name in enumerate(midi_names):
        sorted_idx = np.argsort(-sim_matrix[i])
        best_j = sorted_idx[0]
        best_match_table.add_data(
            midi_name,
            text_names[best_j],
            round(float(sim_matrix[i, best_j]), 4),
            1
        )
    wandb.log({"best_matches": best_match_table})

    # --- Chart: Average similarity per text prompt ---
    avg_per_text = sim_matrix.mean(axis=0)
    text_avg_table = wandb.Table(
        data=[[text_names[j], round(float(avg_per_text[j]), 4)]
              for j in range(len(text_names))],
        columns=["text_prompt", "avg_similarity"]
    )
    wandb.log({
        "avg_similarity_per_prompt": wandb.plot.bar(
            text_avg_table, "text_prompt", "avg_similarity",
            title="Average CLaMP 3 similarity per text prompt"
        )
    })

    # --- Summary metrics ---
    summary = {
        "mean_similarity": round(float(sim_matrix.mean()), 4),
        "max_similarity": round(float(sim_matrix.max()), 4),
        "min_similarity": round(float(sim_matrix.min()), 4),
        "std_similarity": round(float(sim_matrix.std()), 4),
    }

    if extra_metrics:
        summary.update(extra_metrics)

    wandb.log(summary)

    # --- MIDI-to-MIDI similarity (condition comparison) ---
    if midi_midi_sims:
        midi_sim_columns = ["file", "midi_midi_similarity"]
        midi_sim_table = wandb.Table(columns=midi_sim_columns)
        for name, sim in midi_midi_sims.items():
            midi_sim_table.add_data(name, round(sim, 4))
        wandb.log({"midi_midi_similarity": midi_sim_table})

        avg_midi_sim = np.mean(list(midi_midi_sims.values()))
        wandb.log({"avg_midi_midi_similarity": round(float(avg_midi_sim), 4)})

    # --- Heatmap as image ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(text_names) * 1.2),
                                        max(4, len(midi_names) * 0.5)))
        im = ax.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(text_names)))
        ax.set_xticklabels(text_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(midi_names)))
        ax.set_yticklabels(midi_names, fontsize=8)
        ax.set_xlabel("Text prompts")
        ax.set_ylabel("MIDI files")
        ax.set_title(f"CLaMP 3 cosine similarity — {condition_label}")
        plt.colorbar(im, ax=ax, label="Cosine similarity")

        # Add values in cells
        for i in range(len(midi_names)):
            for j in range(len(text_names)):
                val = sim_matrix[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.18 else "black")

        plt.tight_layout()
        heatmap_path = os.path.join(RESULTS_DIR, f"heatmap_{condition_label}.png")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"similarity_heatmap": wandb.Image(heatmap_path)})
        print(f"  Saved heatmap to {heatmap_path}")
    except ImportError:
        print("  matplotlib not installed — skipping heatmap. pip install matplotlib")

    wandb.finish()
    print(f"\nWandb run complete! View at: {run.url}")
    return run


def run_comparison(direct_dir, prompt_dsl_dir, rag_dir, prompts_file):
    """
    Run the three-condition ablation:
      1. Direct controls (no LLM)
      2. Prompt → DSL → controls (mike branch pipeline)
      3. RAG-augmented prompt → DSL → controls

    Each directory should contain MIDI files with matching names
    across conditions (e.g., prompt_01_seed_1.mid).
    """
    # Create prompts as text files from prompts.json
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    prompts_dir = os.path.join(RESULTS_DIR, "prompt_texts")
    os.makedirs(prompts_dir, exist_ok=True)
    for p in prompts:
        txt_path = os.path.join(prompts_dir, f"{p['id']}.txt")
        with open(txt_path, "w") as f:
            f.write(p["text"])

    # Embed prompts
    prompt_embed_dir = os.path.join(CACHE_DIR, "prompt_embeddings")
    embed_texts(prompts_dir, prompt_embed_dir, label="prompts")

    conditions = {
        "direct_controls": direct_dir,
        "prompt_to_dsl": prompt_dsl_dir,
        "rag_augmented": rag_dir,
    }

    all_results = {}

    for cond_name, midi_dir in conditions.items():
        if not midi_dir or not os.path.exists(midi_dir):
            print(f"Skipping {cond_name}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing condition: {cond_name}")
        print(f"{'='*60}")

        mtf_dir = os.path.join(CACHE_DIR, f"mtf_{cond_name}")
        embed_dir = os.path.join(CACHE_DIR, f"embed_{cond_name}")

        # Convert and embed
        convert_and_embed(midi_dir, mtf_dir, embed_dir, label=cond_name)

        # Compute similarity matrix against prompts
        midi_names, text_names, sim_matrix = compute_similarity_matrix(
            embed_dir, prompt_embed_dir
        )

        all_results[cond_name] = {
            "midi_names": midi_names,
            "text_names": text_names,
            "sim_matrix": sim_matrix,
            "embed_dir": embed_dir,
        }

        # Log to wandb
        log_to_wandb(midi_names, text_names, sim_matrix,
                      condition_label=cond_name)

    # Cross-condition MIDI-to-MIDI similarity
    cond_pairs = [
        ("direct_controls", "prompt_to_dsl"),
        ("prompt_to_dsl", "rag_augmented"),
        ("direct_controls", "rag_augmented"),
    ]

    for cond_a, cond_b in cond_pairs:
        if cond_a in all_results and cond_b in all_results:
            print(f"\nComparing {cond_a} vs {cond_b}...")
            sims = compute_midi_midi_similarity(
                all_results[cond_a]["embed_dir"],
                all_results[cond_b]["embed_dir"]
            )
            if sims:
                avg = np.mean(list(sims.values()))
                print(f"  Average MIDI-MIDI similarity: {avg:.4f}")
                print(f"  Files compared: {len(sims)}")

    # Save combined results
    results_path = os.path.join(RESULTS_DIR, "comparison_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results = {}
    for cond_name, res in all_results.items():
        save_results[cond_name] = {
            "midi_names": res["midi_names"],
            "text_names": res["text_names"],
            "sim_matrix": res["sim_matrix"].tolist(),
            "mean_sim": float(res["sim_matrix"].mean()),
            "max_sim": float(res["sim_matrix"].max()),
        }
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved comparison results to {results_path}")


def quick_evaluate(generated_dir, texts_dir=None):
    """
    Quick evaluation: embed generated MIDI files and compare
    against text prompts.
    """
    if texts_dir is None:
        texts_dir = os.path.join(CLAMP3_ROOT, "test_texts")

    if not os.path.exists(texts_dir):
        print(f"Error: texts directory not found: {texts_dir}")
        print("Create test_texts/ with .txt files containing your prompts.")
        return

    mtf_dir = os.path.join(CACHE_DIR, "eval_mtf")
    midi_embed_dir = os.path.join(CACHE_DIR, "eval_midi_embeddings")
    text_embed_dir = os.path.join(CACHE_DIR, "eval_text_embeddings")

    # Convert and embed MIDI
    convert_and_embed(generated_dir, mtf_dir, midi_embed_dir, label="generated")

    # Embed texts
    embed_texts(texts_dir, text_embed_dir, label="prompts")

    # Compute similarity
    midi_names, text_names, sim_matrix = compute_similarity_matrix(
        midi_embed_dir, text_embed_dir
    )

    # Print results
    print(f"\n{'='*60}")
    print("SIMILARITY MATRIX")
    print(f"{'='*60}")
    header = f"{'MIDI file':<30}" + "".join(f"{t:<25}" for t in text_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(midi_names):
        row = f"{name:<30}" + "".join(f"{sim_matrix[i,j]:<25.4f}"
                                       for j in range(len(text_names)))
        print(row)

    print(f"\nMean similarity: {sim_matrix.mean():.4f}")
    print(f"Max similarity:  {sim_matrix.max():.4f}")

    # Log to wandb
    print("\nLogging to wandb...")
    log_to_wandb(midi_names, text_names, sim_matrix,
                  condition_label="quick_eval")


def generate_prompts_template():
    
    prompts = [
        # --- Unambiguous single-attribute (p01-p06) ---
        {"id": "p01", "text": "fast energetic piano with high density and bright chords",
         "controls": {"instrument": "piano", "mood": "energetic", "complexity": "high",
                      "polyphony_level": "high", "density_level": "high", "register": "high",
                      "note_duration_level": "short", "genre": "unknown"}},
        {"id": "p02", "text": "slow calm piano with sparse notes and low register",
         "controls": {"instrument": "piano", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "low",
                      "note_duration_level": "long", "genre": "ambient"}},
        {"id": "p03", "text": "bright happy melody with high register",
         "controls": {"instrument": "piano", "mood": "happy", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "medium", "register": "high",
                      "note_duration_level": "medium", "genre": "pop"}},
        {"id": "p04", "text": "dark sad piano with deep low notes and sustained legato",
         "controls": {"instrument": "piano", "mood": "sad", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "low", "register": "low",
                      "note_duration_level": "long", "genre": "classical"}},
        {"id": "p05", "text": "dense complex polyphonic piano with layered chords",
         "controls": {"instrument": "piano", "mood": "neutral", "complexity": "high",
                      "polyphony_level": "high", "density_level": "high", "register": "medium",
                      "note_duration_level": "medium", "genre": "classical"}},
        {"id": "p06", "text": "simple solo monophonic melody with long sustained notes",
         "controls": {"instrument": "piano", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "medium",
                      "note_duration_level": "long", "genre": "unknown"}},

        # --- Multi-attribute (p07-p10) ---
        {"id": "p07", "text": "calm ambient with sparse polyphony and long sustained pads",
         "controls": {"instrument": "synth", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "medium",
                      "note_duration_level": "long", "genre": "ambient"}},
        {"id": "p08", "text": "fast energetic electronic with dense driving short staccato notes",
         "controls": {"instrument": "synth", "mood": "energetic", "complexity": "high",
                      "polyphony_level": "medium", "density_level": "high", "register": "medium",
                      "note_duration_level": "short", "genre": "electronic"}},
        {"id": "p09", "text": "jazz piano with medium complexity and relaxing calm mood",
         "controls": {"instrument": "piano", "mood": "calm", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "medium", "register": "medium",
                      "note_duration_level": "medium", "genre": "jazz"}},
        {"id": "p10", "text": "sparse bright airy high register with calm relaxing mood",
         "controls": {"instrument": "piano", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "high",
                      "note_duration_level": "long", "genre": "ambient"}},

        # --- Ambiguous natural language (p11-p15) ---
        {"id": "p11", "text": "melancholic jazz piano for a rainy evening",
         "controls": {"instrument": "piano", "mood": "sad", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "medium", "register": "medium",
                      "note_duration_level": "medium", "genre": "jazz"}},
        {"id": "p12", "text": "gentle folk guitar ballad with calm acoustic warmth",
         "controls": {"instrument": "guitar", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "medium",
                      "note_duration_level": "medium", "genre": "folk"}},
        {"id": "p13", "text": "dramatic cinematic orchestral strings with dense layered climax",
         "controls": {"instrument": "strings", "mood": "dramatic", "complexity": "high",
                      "polyphony_level": "high", "density_level": "high", "register": "medium",
                      "note_duration_level": "medium", "genre": "cinematic"}},
        {"id": "p14", "text": "minimal ambient synth drone with sustained long notes",
         "controls": {"instrument": "synth", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "medium",
                      "note_duration_level": "long", "genre": "ambient"}},
        {"id": "p15", "text": "uplifting bright happy piano bouncing between registers",
         "controls": {"instrument": "piano", "mood": "happy", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "medium", "register": "high",
                      "note_duration_level": "short", "genre": "pop"}},

        # --- Edge cases (p16-p20) ---
        {"id": "p16", "text": "tense dark dense complex layered synth",
         "controls": {"instrument": "synth", "mood": "dark", "complexity": "high",
                      "polyphony_level": "high", "density_level": "high", "register": "low",
                      "note_duration_level": "medium", "genre": "electronic"}},
        {"id": "p17", "text": "simple solo piano with sparse calm sustained notes",
         "controls": {"instrument": "piano", "mood": "calm", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "medium",
                      "note_duration_level": "long", "genre": "unknown"}},
        {"id": "p18", "text": "deep bass-heavy low register with slow driving rhythm",
         "controls": {"instrument": "bass", "mood": "dark", "complexity": "medium",
                      "polyphony_level": "low", "density_level": "medium", "register": "low",
                      "note_duration_level": "medium", "genre": "electronic"}},
        {"id": "p19", "text": "bright staccato short plucky piano arpeggios climbing upward",
         "controls": {"instrument": "piano", "mood": "happy", "complexity": "medium",
                      "polyphony_level": "low", "density_level": "high", "register": "high",
                      "note_duration_level": "short", "genre": "unknown"}},
        {"id": "p20", "text": "rich polyphonic classical piano with layered intricate chords and legato",
         "controls": {"instrument": "piano", "mood": "neutral", "complexity": "high",
                      "polyphony_level": "high", "density_level": "medium", "register": "medium",
                      "note_duration_level": "long", "genre": "classical"}},

        # --- Non-piano instruments (p21-p25) ---
        {"id": "p21", "text": "energetic rock guitar with fast dense driving rhythm",
         "controls": {"instrument": "guitar", "mood": "energetic", "complexity": "high",
                      "polyphony_level": "medium", "density_level": "high", "register": "medium",
                      "note_duration_level": "short", "genre": "rock"}},
        {"id": "p22", "text": "dark dramatic organ with sustained low register chords",
         "controls": {"instrument": "organ", "mood": "dramatic", "complexity": "medium",
                      "polyphony_level": "high", "density_level": "medium", "register": "low",
                      "note_duration_level": "long", "genre": "classical"}},
        {"id": "p23", "text": "happy uplifting pop synth with bright high register melody",
         "controls": {"instrument": "synth", "mood": "happy", "complexity": "medium",
                      "polyphony_level": "medium", "density_level": "medium", "register": "high",
                      "note_duration_level": "medium", "genre": "pop"}},
        {"id": "p24", "text": "slow melancholic cello solo with long legato notes",
         "controls": {"instrument": "strings", "mood": "sad", "complexity": "low",
                      "polyphony_level": "low", "density_level": "low", "register": "low",
                      "note_duration_level": "long", "genre": "classical"}},
        {"id": "p25", "text": "energetic trance synth with fast dense driving electronic beat",
         "controls": {"instrument": "synth", "mood": "energetic", "complexity": "high",
                      "polyphony_level": "medium", "density_level": "high", "register": "medium",
                      "note_duration_level": "short", "genre": "trance"}},
    ]

    path = os.path.join(CLAMP3_ROOT, "prompts.json")
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} prompts template at {path}")
    print("Each prompt has a 'text' field (natural language) and a 'controls' field")
    print("(the direct MIDI-GPT parameters for condition 1).")
    print("\nEdit this file to match your DSL variable space and add more prompts.")


def main():
    parser = argparse.ArgumentParser(description="CLaMP 3 Evaluation Pipeline")
    parser.add_argument("--download", action="store_true",
                        help="Download MIDICaps subset")
    parser.add_argument("--subset_size", type=int, default=200,
                        help="Number of MIDICaps files to download")
    parser.add_argument("--evaluate", action="store_true",
                        help="Quick evaluation on generated MIDI files")
    parser.add_argument("--generated_dir", type=str,
                        help="Directory containing generated MIDI files")
    parser.add_argument("--texts_dir", type=str, default=None,
                        help="Directory containing text prompt files")
    parser.add_argument("--compare", action="store_true",
                        help="Run three-condition comparison")
    parser.add_argument("--direct_controls_dir", type=str)
    parser.add_argument("--prompt_to_dsl_dir", type=str)
    parser.add_argument("--rag_augmented_dir", type=str)
    parser.add_argument("--prompts_file", type=str, default="prompts.json")
    parser.add_argument("--generate_prompts", action="store_true",
                        help="Generate prompts.json template")

    args = parser.parse_args()

    if args.generate_prompts:
        generate_prompts_template()
    elif args.download:
        download_midicaps_subset(args.subset_size)
    elif args.evaluate:
        if not args.generated_dir:
            print("Error: --generated_dir required for --evaluate")
            return
        quick_evaluate(args.generated_dir, args.texts_dir)
    elif args.compare:
        run_comparison(
            args.direct_controls_dir,
            args.prompt_to_dsl_dir,
            args.rag_augmented_dir,
            args.prompts_file
        )
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python eval_pipeline.py --generate_prompts")
        print("  2. python eval_pipeline.py --download --subset_size 200")
        print("  3. python eval_pipeline.py --evaluate --generated_dir test_midi")


if __name__ == "__main__":
    main()
