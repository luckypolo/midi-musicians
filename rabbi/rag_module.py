"""
RAG Module for Prompt-Augmented MIDI Generation
=================================================
Retrieves similar MIDI files from a FAISS index built on MIDICaps,
injects their captions into the LLM prompt as few-shot context,
and produces DSL controls via the existing prompt_to_controls pipeline.

This module sits between the user prompt and the LLM call.
It does NOT modify the canonicalization or MIDI-GPT bridge — it only
enriches the prompt that gets sent to the LLM (or heuristic parser).

Usage:
  # As a module (imported by batch_generate.py or similar):
  from rag_module import RAGRetriever
  rag = RAGRetriever(index_dir="faiss_index", clamp3_root="path/to/clamp3")
  augmented_prompt, retrieved = rag.augment_prompt("fast energetic piano", top_k=3)

  # Standalone test:
  python rag_module.py --query "calm ambient piano" --top_k 3
  python rag_module.py --prompts prompts.json --top_k 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss


@dataclass
class RetrievalResult:
    """A single retrieved MIDI file with its caption and similarity score."""
    filename: str
    caption: str
    score: float
    index: int


class RAGRetriever:
    """
    Retrieves similar MIDI captions from a FAISS index
    and builds augmented prompts for the LLM.
    """

    def __init__(
        self,
        index_dir: str = "faiss_index",
        clamp3_root: str = None,
        embedding_dim: int = 768,
    ):
        """
        Args:
            index_dir: Directory containing midicaps.index and midicaps_metadata.json
            clamp3_root: Path to clamp3 repo (needed for embedding text queries).
                         If None, tries to auto-detect from common locations.
            embedding_dim: CLaMP 3 embedding dimension (768 for C2)
        """
        self.index_dir = os.path.abspath(index_dir)
        self.embedding_dim = embedding_dim

        # Load FAISS index
        index_path = os.path.join(self.index_dir, "midicaps.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"[RAG] Loaded FAISS index with {self.index.ntotal} vectors")

        # Load metadata
        metadata_path = os.path.join(self.index_dir, "midicaps_metadata.json")
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Find clamp3 root
        if clamp3_root:
            self.clamp3_root = os.path.abspath(clamp3_root)
        else:
            # Try common relative paths
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "clamp3"),
                os.path.join(os.path.dirname(__file__), "clamp3"),
                os.path.expanduser("~/clamp3"),
            ]
            for c in candidates:
                if os.path.exists(os.path.join(c, "code", "extract_clamp3.py")):
                    self.clamp3_root = os.path.abspath(c)
                    break
            else:
                raise FileNotFoundError(
                    "Could not find clamp3 repo. Pass --clamp3_root explicitly."
                )
        print(f"[RAG] Using CLaMP 3 from {self.clamp3_root}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query using CLaMP 3.
        Creates a temp file, runs extract_clamp3.py, returns the embedding.
        """
        temp_dir = os.path.join(self.clamp3_root, "_rag_temp_query")
        temp_emb_dir = os.path.join(self.clamp3_root, "_rag_temp_emb")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(temp_emb_dir, exist_ok=True)

        try:
            # Write query text
            query_path = os.path.join(temp_dir, "query.txt")
            with open(query_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Run CLaMP 3 embedding extraction
            code_dir = os.path.join(self.clamp3_root, "code")
            cmd = [
                sys.executable, "extract_clamp3.py",
                os.path.abspath(temp_dir),
                os.path.abspath(temp_emb_dir),
                "--get_global"
            ]
            subprocess.run(
                cmd, cwd=code_dir, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Load the embedding
            npy_files = list(Path(temp_emb_dir).glob("*.npy"))
            if not npy_files:
                raise RuntimeError("CLaMP 3 embedding extraction produced no output")

            emb = np.load(npy_files[0]).flatten().astype(np.float32)
            return emb

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(temp_emb_dir, ignore_errors=True)

    def retrieve(self, query_text: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Retrieve top-k most similar MIDI files for a text query.
        """
        emb = self.embed_text(query_text)
        emb = emb.reshape(1, -1)
        faiss.normalize_L2(emb)

        scores, indices = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(RetrievalResult(
                filename=meta["filename"],
                caption=meta.get("caption", ""),
                score=float(score),
                index=int(idx),
            ))

        return results

    def build_augmented_prompt(
        self,
        user_prompt: str,
        retrieved: list[RetrievalResult],
        mode: str = "caption_examples",
    ) -> str:
        """
        Build an augmented prompt that includes retrieved captions as context.

        Args:
            user_prompt: The original user text prompt
            retrieved: List of RetrievalResult from retrieve()
            mode: How to format the context:
                - "caption_examples": Include captions as reference examples
                - "caption_with_controls": Include captions + extracted DSL controls

        Returns:
            Augmented prompt string for the LLM
        """
        if mode == "caption_examples":
            examples = []
            for i, r in enumerate(retrieved, 1):
                if r.caption:
                    examples.append(f"Example {i} (similarity: {r.score:.3f}):\n{r.caption}")

            if not examples:
                return user_prompt

            context = "\n\n".join(examples)
            augmented = (
                f"Here are descriptions of real MIDI files that are musically similar "
                f"to what the user is asking for. Use these as reference for the style, "
                f"mood, instrumentation, and musical characteristics when generating controls.\n\n"
                f"{context}\n\n"
                f"Now, given the following user request, produce the appropriate DSL controls:\n"
                f"User request: {user_prompt}"
            )

        elif mode == "caption_with_controls":
            # Parse each caption through parse_prompt to show example control mappings
            try:
                from prompt_to_controls import parse_prompt
                examples = []
                for i, r in enumerate(retrieved, 1):
                    if r.caption:
                        controls = parse_prompt(r.caption)
                        examples.append(
                            f"Example {i} (similarity: {r.score:.3f}):\n"
                            f"  Caption: {r.caption[:200]}\n"
                            f"  Controls: mood={controls.mood}, density={controls.density_level}, "
                            f"register={controls.register}, polyphony={controls.polyphony_level}, "
                            f"instrument={controls.instrument}, genre={controls.genre}"
                        )

                context = "\n\n".join(examples)
                augmented = (
                    f"Here are descriptions of similar MIDI files with their extracted controls. "
                    f"Use these as reference when choosing control values.\n\n"
                    f"{context}\n\n"
                    f"User request: {user_prompt}\n"
                    f"Produce the DSL controls for this request."
                )
            except ImportError:
                # Fallback if prompt_to_controls not available
                return self.build_augmented_prompt(user_prompt, retrieved, mode="caption_examples")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return augmented

    def augment_prompt(
        self,
        user_prompt: str,
        top_k: int = 3,
        mode: str = "caption_examples",
    ) -> tuple[str, list[RetrievalResult]]:
        """
        Full pipeline: retrieve similar MIDI files and build augmented prompt.

        Returns:
            (augmented_prompt, list_of_retrieved_results)
        """
        retrieved = self.retrieve(user_prompt, top_k=top_k)
        augmented = self.build_augmented_prompt(user_prompt, retrieved, mode=mode)
        return augmented, retrieved



def generate_condition3_controls(
    prompts_file: str,
    index_dir: str = "faiss_index",
    clamp3_root: str = None,
    top_k: int = 3,
    mode: str = "caption_with_controls",
    output_file: str = None,
):
    """
    For each prompt, retrieve similar MIDI files and produce
    RAG-augmented DSL controls using parse_prompt on the augmented text.

    This gives you condition 3 controls that can be compared against
    condition 1 (direct) and condition 2 (heuristic parse only).
    """
    from prompt_to_controls import parse_prompt
    from dataclasses import asdict

    rag = RAGRetriever(index_dir=index_dir, clamp3_root=clamp3_root)

    with open(prompts_file) as f:
        prompts = json.load(f)

    results = []

    for prompt in prompts:
        pid = prompt["id"]
        text = prompt["text"]

        print(f"\n{'='*60}")
        print(f"[{pid}] {text}")

        # Retrieve similar MIDI files
        retrieved = rag.retrieve(text, top_k=top_k)

        # Show what was retrieved
        for r in retrieved:
            print(f"  Retrieved: {r.caption[:70]}... (score={r.score:.3f})")

        # Strategy: Use retrieved captions to inform control selection
        # We parse the original prompt AND each retrieved caption,
        # then use a weighted vote to decide final controls
        original_controls = parse_prompt(text)
        retrieved_controls = [parse_prompt(r.caption) for r in retrieved if r.caption]

        # Weighted voting: original prompt gets weight 1.0,
        # each retrieved example gets weight proportional to similarity score
        def weighted_vote(field: str, original_value: str, retrieved_ctrls, retrieved_results):
            """Vote on a control value using similarity-weighted retrieved examples."""
            votes = {original_value: 1.0}  # Original prompt gets weight 1.0

            for ctrl, result in zip(retrieved_ctrls, retrieved_results):
                value = getattr(ctrl, field)
                weight = result.score  # Use similarity as weight
                votes[value] = votes.get(value, 0) + weight

            # Return the value with highest total weight
            winner = max(votes, key=votes.get)
            return winner

        # Fields to vote on
        vote_fields = [
            "instrument", "mood", "complexity", "polyphony_level",
            "density_level", "register", "note_duration_level", "genre"
        ]

        final_controls = {}
        for field in vote_fields:
            original_val = getattr(original_controls, field)
            final_val = weighted_vote(field, original_val, retrieved_controls, retrieved)
            final_controls[field] = final_val

        # Compare with original parse
        original_dict = {f: getattr(original_controls, f) for f in vote_fields}
        changes = {f: (original_dict[f], final_controls[f])
                   for f in vote_fields if original_dict[f] != final_controls[f]}

        print(f"  Original controls: {original_dict}")
        print(f"  RAG-augmented controls: {final_controls}")
        if changes:
            print(f"  Changes: {changes}")
        else:
            print(f"  No changes from RAG augmentation")

        results.append({
            "id": pid,
            "text": text,
            "original_controls": original_dict,
            "rag_controls": final_controls,
            "changes": {k: list(v) for k, v in changes.items()},
            "retrieved": [
                {"filename": r.filename, "caption": r.caption[:200], "score": r.score}
                for r in retrieved
            ],
        })

    # Save results
    if output_file is None:
        output_file = "condition3_rag_controls.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved RAG-augmented controls to {output_file}")

    # Summary
    total_changes = sum(len(r["changes"]) for r in results)
    prompts_changed = sum(1 for r in results if r["changes"])
    print(f"\nSummary:")
    print(f"  Total prompts: {len(results)}")
    print(f"  Prompts with control changes: {prompts_changed}/{len(results)}")
    print(f"  Total field changes: {total_changes}")

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG module for MIDI generation")
    parser.add_argument("--query", help="Single text query to test")
    parser.add_argument("--prompts", help="Prompts JSON file for batch processing")
    parser.add_argument("--index_dir", default="faiss_index")
    parser.add_argument("--clamp3_root", default=None,
                        help="Path to clamp3 repo root")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--mode", default="caption_with_controls",
                        choices=["caption_examples", "caption_with_controls"])
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.query:
        # Single query test
        rag = RAGRetriever(index_dir=args.index_dir, clamp3_root=args.clamp3_root)
        augmented, retrieved = rag.augment_prompt(args.query, top_k=args.top_k, mode=args.mode)

        print(f"\nOriginal query: {args.query}")
        print(f"\nRetrieved {len(retrieved)} similar MIDI files:")
        for r in retrieved:
            print(f"  [{r.score:.3f}] {r.caption[:80]}...")

        print(f"\nAugmented prompt:")
        print("-" * 60)
        print(augmented)
        print("-" * 60)

    elif args.prompts:
        # Batch: generate condition 3 controls for all prompts
        generate_condition3_controls(
            prompts_file=args.prompts,
            index_dir=args.index_dir,
            clamp3_root=args.clamp3_root,
            top_k=args.top_k,
            mode=args.mode,
            output_file=args.output,
        )

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python rag_module.py --query 'calm ambient piano' --clamp3_root ../clamp3")
        print("  python rag_module.py --prompts prompts.json --clamp3_root ../clamp3")


if __name__ == "__main__":
    main()
