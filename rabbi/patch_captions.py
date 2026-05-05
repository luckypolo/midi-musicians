"""
Patch FAISS metadata with captions from MIDICaps train.json
============================================================
Run this from your clamp3/ directory to fix the empty captions.

Usage:
  python patch_captions.py --train_json midicaps_subset/train.json --metadata faiss_index/midicaps_metadata.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True, help="Path to MIDICaps train.json")
    parser.add_argument("--metadata", default="faiss_index/midicaps_metadata.json")
    args = parser.parse_args()

    # Load train.json (JSONL format: one JSON object per line)
    print(f"Loading captions from {args.train_json}...")
    caption_lookup = {}

    with open(args.train_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Extract caption and build lookup by hash
                caption = item.get("caption", "")
                # Try multiple filename keys
                for key in ["location", "filename", "file", "path"]:
                    if key in item:
                        # Extract just the hash stem from any path
                        stem = Path(str(item[key])).stem
                        caption_lookup[stem] = caption
                        break
                else:
                    # If no filename key, try using the whole item
                    # Some formats use the hash as a top-level key
                    pass
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(caption_lookup)} captions")

    # Show sample entries to debug the format
    sample_keys = list(caption_lookup.keys())[:3]
    print(f"  Sample keys: {sample_keys}")
    for k in sample_keys:
        print(f"    {k}: {caption_lookup[k][:80]}...")

    # Load metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    print(f"\nPatching {len(metadata)} metadata entries...")

    matched = 0
    for entry in metadata:
        filename = entry["filename"]
        # Try direct match
        if filename in caption_lookup:
            entry["caption"] = caption_lookup[filename]
            matched += 1
        else:
            # Try stripping common prefixes
            for prefix in ["mid-", "midi-", ""]:
                clean = filename.replace(prefix, "")
                if clean in caption_lookup:
                    entry["caption"] = caption_lookup[clean]
                    matched += 1
                    break

    print(f"  Matched {matched}/{len(metadata)} entries with captions")

    # Save patched metadata
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved patched metadata to {args.metadata}")

    # Show some results
    print("\nSample patched entries:")
    for entry in metadata[:5]:
        caption_preview = entry["caption"][:80] + "..." if len(entry["caption"]) > 80 else entry["caption"]
        print(f"  {entry['filename']}: {caption_preview if caption_preview else '(no caption)'}")


if __name__ == "__main__":
    main()
