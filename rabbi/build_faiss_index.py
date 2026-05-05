"""
Build FAISS Index from MIDICaps CLaMP 3 Embeddings
====================================================
This script:
  1. Converts MIDICaps MIDI files to MTF format
  2. Extracts CLaMP 3 embeddings (if not already done)
  3. Builds a FAISS index for fast similarity search
  4. Saves the index + metadata mapping to disk
  5. Tests retrieval with sample prompts

Place this script in your clamp3/ repo root.

Prerequisites:
  pip install faiss-cpu

Usage:
  # Full pipeline: convert, embed, index, test
  python build_faiss_index.py --midi_dir midicaps_subset/midi --subset_meta midicaps_subset/subset_meta.json

  # If you already have embeddings, skip conversion:
  python build_faiss_index.py --embeddings_dir midicaps_subset/embeddings --subset_meta midicaps_subset/subset_meta.json

  # Test retrieval on an existing index:
  python build_faiss_index.py --test_only --query "fast energetic piano"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss


CLAMP3_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(CLAMP3_ROOT, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "midicaps.index")
METADATA_PATH = os.path.join(INDEX_DIR, "midicaps_metadata.json")
EMBEDDING_DIM = 768  # CLaMP 3 embedding dimension


def convert_midi_to_mtf(midi_dir: str, mtf_dir: str):
    """Convert all MIDI files in midi_dir to MTF format."""
    os.makedirs(mtf_dir, exist_ok=True)

    # Check if MTF files already exist
    existing_mtf = list(Path(mtf_dir).glob("*.mtf"))
    existing_midi = list(Path(midi_dir).glob("*.mid")) + list(Path(midi_dir).glob("*.midi"))

    if len(existing_mtf) >= len(existing_midi) and len(existing_mtf) > 0:
        print(f"  MTF files already exist ({len(existing_mtf)} files). Skipping conversion.")
        return

    print(f"  Converting {len(existing_midi)} MIDI files to MTF...")
    preprocessing_dir = os.path.join(CLAMP3_ROOT, "preprocessing", "midi")
    cmd = [
        sys.executable, "batch_midi2mtf.py",
        os.path.abspath(midi_dir),
        os.path.abspath(mtf_dir),
        "--m3_compatible"
    ]
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)

    result_count = len(list(Path(mtf_dir).glob("*.mtf")))
    print(f"  Created {result_count} MTF files")


def extract_embeddings(mtf_dir: str, embeddings_dir: str):
    """Extract CLaMP 3 global embeddings for all MTF files."""
    os.makedirs(embeddings_dir, exist_ok=True)

    existing_npy = list(Path(embeddings_dir).glob("*.npy"))
    existing_mtf = list(Path(mtf_dir).glob("*.mtf"))

    if len(existing_npy) >= len(existing_mtf) and len(existing_npy) > 0:
        print(f"  Embeddings already exist ({len(existing_npy)} files). Skipping extraction.")
        return

    print(f"  Extracting CLaMP 3 embeddings for {len(existing_mtf)} files...")
    code_dir = os.path.join(CLAMP3_ROOT, "code")
    cmd = [
        sys.executable, "extract_clamp3.py",
        os.path.abspath(mtf_dir),
        os.path.abspath(embeddings_dir),
        "--get_global"
    ]
    subprocess.run(cmd, cwd=code_dir, check=True)

    result_count = len(list(Path(embeddings_dir).glob("*.npy")))
    print(f"  Created {result_count} embedding files")


def build_index(embeddings_dir: str, metadata_path: str = None):
    """
    Build a FAISS index from .npy embedding files.

    Uses IndexFlatIP (inner product) since CLaMP 3 embeddings
    are L2-normalized, making inner product = cosine similarity.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load all embeddings
    npy_files = sorted(Path(embeddings_dir).glob("*.npy"))
    if not npy_files:
        print(f"ERROR: No .npy files found in {embeddings_dir}")
        return None

    print(f"\nLoading {len(npy_files)} embeddings...")

    embeddings = []
    file_stems = []

    for f in npy_files:
        emb = np.load(f).flatten().astype(np.float32)
        # Strip the "mid-" prefix CLaMP 3 adds to filenames
        stem = f.stem
        if stem.startswith("mid-"):
            stem = stem[4:]
        embeddings.append(emb)
        file_stems.append(stem)

    embeddings_matrix = np.stack(embeddings)
    print(f"  Embedding matrix shape: {embeddings_matrix.shape}")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings_matrix)

    # Build index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_matrix)
    print(f"  Index built with {index.ntotal} vectors")

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"  Index saved to {INDEX_PATH}")

    # Build metadata mapping: index position -> file info
    # Load captions from subset_meta.json if available
    captions = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path) as f:
            subset_meta = json.load(f)
        # Build lookup from safe_name to caption
        for filename, info in subset_meta.items():
            safe = info.get("safe_name", Path(filename).stem)
            captions[safe] = info.get("caption", "")

    # Also check for individual .txt caption files
    texts_dir = os.path.join(os.path.dirname(embeddings_dir), "texts")
    if os.path.exists(texts_dir):
        for txt_file in Path(texts_dir).glob("*.txt"):
            with open(txt_file) as f:
                captions[txt_file.stem] = f.read().strip()

    metadata = []
    for i, stem in enumerate(file_stems):
        entry = {
            "index": i,
            "filename": stem,
            "caption": captions.get(stem, ""),
        }
        metadata.append(entry)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {METADATA_PATH}")
    print(f"  Files with captions: {sum(1 for m in metadata if m['caption'])}/{len(metadata)}")

    return index, metadata


def load_index():
    """Load existing FAISS index and metadata."""
    if not os.path.exists(INDEX_PATH):
        print(f"ERROR: No index found at {INDEX_PATH}")
        print("Run with --midi_dir to build the index first.")
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata


def embed_text_query(text: str) -> np.ndarray:
    """
    Embed a text query using CLaMP 3.
    Creates a temp .txt file, runs extract_clamp3.py, loads the .npy result.
    """
    temp_dir = os.path.join(CLAMP3_ROOT, "temp_query")
    temp_emb_dir = os.path.join(CLAMP3_ROOT, "temp_query_emb")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_emb_dir, exist_ok=True)

    # Write query to temp file
    query_path = os.path.join(temp_dir, "query.txt")
    with open(query_path, "w") as f:
        f.write(text)

    # Extract embedding
    code_dir = os.path.join(CLAMP3_ROOT, "code")
    cmd = [
        sys.executable, "extract_clamp3.py",
        os.path.abspath(temp_dir),
        os.path.abspath(temp_emb_dir),
        "--get_global"
    ]
    subprocess.run(cmd, cwd=code_dir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load embedding
    npy_files = list(Path(temp_emb_dir).glob("*.npy"))
    if not npy_files:
        print("ERROR: Failed to embed query text")
        return None

    emb = np.load(npy_files[0]).flatten().astype(np.float32)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(temp_emb_dir, ignore_errors=True)

    return emb


def search(query_text: str, index, metadata, top_k: int = 5):
    """
    Search the FAISS index with a text query.
    Returns the top-k most similar MIDI files with their captions.
    """
    print(f"\nQuery: \"{query_text}\"")
    print("Embedding query...")

    emb = embed_text_query(query_text)
    if emb is None:
        return []

    # Normalize for cosine similarity
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)

    # Search
    scores, indices = index.search(emb, top_k)

    results = []
    print(f"\nTop {top_k} results:")
    print(f"{'Rank':<6} {'Score':<10} {'Filename':<35} {'Caption':<60}")
    print("-" * 110)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        caption_preview = meta["caption"][:57] + "..." if len(meta["caption"]) > 57 else meta["caption"]
        print(f"{rank:<6} {score:<10.4f} {meta['filename']:<35} {caption_preview}")
        results.append({
            "rank": rank,
            "score": float(score),
            "filename": meta["filename"],
            "caption": meta["caption"],
            "index": int(idx),
        })

    return results


def test_with_prompts(index, metadata, prompts_file: str = "prompts.json", top_k: int = 3):
    """Run retrieval for all prompts and show results."""
    if not os.path.exists(prompts_file):
        # Try alternate names
        for alt in ["prompts_crafted.json", "prompts.json"]:
            if os.path.exists(alt):
                prompts_file = alt
                break
        else:
            print(f"No prompts file found. Skipping batch test.")
            return

    with open(prompts_file) as f:
        prompts = json.load(f)

    print(f"\n{'='*80}")
    print(f"BATCH RETRIEVAL TEST — {len(prompts)} prompts, top-{top_k} results each")
    print(f"{'='*80}")

    all_results = {}
    for prompt in prompts:
        pid = prompt["id"]
        text = prompt["text"]
        print(f"\n--- {pid}: {text} ---")
        results = search(text, index, metadata, top_k=top_k)
        all_results[pid] = results

    # Save results
    results_path = os.path.join(INDEX_DIR, "retrieval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBatch results saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from MIDICaps CLaMP 3 embeddings")
    parser.add_argument("--midi_dir", help="Directory with MIDICaps MIDI files")
    parser.add_argument("--embeddings_dir", help="Directory with pre-computed .npy embeddings (skip conversion)")
    parser.add_argument("--subset_meta", help="Path to subset_meta.json with captions")
    parser.add_argument("--test_only", action="store_true", help="Only test retrieval on existing index")
    parser.add_argument("--query", help="Single text query to test retrieval")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--prompts", default="prompts.json", help="Prompts file for batch testing")
    parser.add_argument("--batch_test", action="store_true", help="Run retrieval on all prompts")
    args = parser.parse_args()

    if args.test_only or (args.query and not args.midi_dir and not args.embeddings_dir):
        # Load existing index and test
        index, metadata = load_index()
        if index is None:
            return

        if args.query:
            search(args.query, index, metadata, top_k=args.top_k)
        elif args.batch_test:
            test_with_prompts(index, metadata, args.prompts, top_k=args.top_k)
        else:
            # Default test queries
            test_queries = [
                "fast energetic piano with high density",
                "slow calm ambient with sparse notes",
                "dark sad melody with low register",
                "bright happy uplifting pop",
            ]
            for q in test_queries:
                search(q, index, metadata, top_k=args.top_k)
        return

    if args.embeddings_dir:
        # Skip conversion, go straight to indexing
        print("Using pre-computed embeddings...")
        index, metadata = build_index(args.embeddings_dir, args.subset_meta)
    elif args.midi_dir:
        # Full pipeline
        midi_dir = os.path.abspath(args.midi_dir)
        base_dir = os.path.dirname(midi_dir)
        mtf_dir = os.path.join(base_dir, "mtf")
        emb_dir = os.path.join(base_dir, "embeddings")

        print(f"MIDI directory: {midi_dir}")
        midi_count = len(list(Path(midi_dir).glob("*.mid")) + list(Path(midi_dir).glob("*.midi")))
        print(f"Found {midi_count} MIDI files")

        # Step 1: Convert MIDI to MTF
        print("\n[Step 1/3] Converting MIDI to MTF...")
        convert_midi_to_mtf(midi_dir, mtf_dir)

        # Step 2: Extract CLaMP 3 embeddings
        print("\n[Step 2/3] Extracting CLaMP 3 embeddings...")
        extract_embeddings(mtf_dir, emb_dir)

        # Step 3: Build FAISS index
        print("\n[Step 3/3] Building FAISS index...")
        index, metadata = build_index(emb_dir, args.subset_meta)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Build index from MIDI files:")
        print("  python build_faiss_index.py --midi_dir midicaps_subset/midi --subset_meta midicaps_subset/subset_meta.json")
        print()
        print("  # Test single query:")
        print("  python build_faiss_index.py --test_only --query \"calm ambient piano\"")
        print()
        print("  # Test all prompts:")
        print("  python build_faiss_index.py --test_only --batch_test --prompts prompts.json")
        return

    if index is None:
        return

    # Run test queries
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)

    test_queries = [
        "fast energetic piano with high density and bright chords",
        "slow calm piano with sparse notes and low register",
        "dark sad melody with deep low notes",
        "dense complex polyphonic piano with layered chords",
    ]

    for q in test_queries:
        search(q, index, metadata, top_k=3)

    # Batch test if prompts file exists
    if os.path.exists(args.prompts):
        print(f"\nRun batch test on all prompts with:")
        print(f"  python build_faiss_index.py --test_only --batch_test --prompts {args.prompts}")


if __name__ == "__main__":
    main()
