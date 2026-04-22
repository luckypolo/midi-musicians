# MIDI2Tags Testing Guide

This guide explains how to reproduce the simplified MIDI2Text/MIDI2Tags pipeline.

The goal is not full caption generation. The model predicts a small set of interpretable tags from MIDI:

- `mood`
- `density_level`
- `polyphony_level`
- `note_duration_level`
- `register`

## 1. Requirements

Install the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Or install only the MIDI2Tags dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-midi2tags.txt
```

The important packages for this pipeline are:

- `miditok`
- `symusic`
- `torch`
- `datasets`

## 2. Download MIDI Data

MidiCaps gives captions and metadata paths, but not the MIDI bytes. Download LMD-full:

```powershell
.\.venv\Scripts\python.exe .\download_lmd_full.py --output-dir .\data --yes
```

Expected output folder:

```text
data/lmd_full
```

## 3. Prepare MIDI2Tags Dataset

Build a small single-track dataset from MidiCaps metadata and LMD-full MIDI files:

```powershell
.\.venv\Scripts\python.exe .\prepare_midi2tag_dataset.py --metadata-json .\data\control_adapter_augmented\train.json --output-dir .\data\midi2tag --max-examples 1000 --max-seq-len 1024
```

Expected outputs:

```text
data/midi2tag/train.json
data/midi2tag/val.json
data/midi2tag/test.json
data/midi2tag/dataset_summary.json
```

Current local run produced:

```text
509 examples
369 train / 64 val / 76 test
```

## 4. Train Model

Train the lightweight REMI-token classifier:

```powershell
.\.venv\Scripts\python.exe .\train_midi2tag.py --train .\data\midi2tag\train.json --val .\data\midi2tag\val.json --output-dir .\artifacts\midi2tag --epochs 30 --batch-size 32 --max-train-steps 1000 --embedding-dim 128 --hidden-dim 128 --max-seq-len 1024
```

Expected outputs:

```text
artifacts/midi2tag/midi2tag_model.pt
artifacts/midi2tag/metadata.json
artifacts/midi2tag/metrics.json
```

## 5. Evaluate Test Split

Run held-out evaluation:

```powershell
.\.venv\Scripts\python.exe .\evaluate_midi2tag.py --model-dir .\artifacts\midi2tag --data .\data\midi2tag\test.json --output .\outputs\midi2tag_eval.json
```

Current local test result:

```text
mean_accuracy: 0.7105
mood_accuracy: 0.5263
density_level_accuracy: 0.7105
polyphony_level_accuracy: 0.7368
note_duration_level_accuracy: 0.6053
register_accuracy: 0.9737
```

## 6. Run Inference On Generated MIDI

Hybrid MIDI-GPT outputs:

```powershell
.\.venv\Scripts\python.exe .\infer_midi2tag.py .\outputs\next_phase_compare\hybrid\generated --model-dir .\artifacts\midi2tag --output .\outputs\midi2tag_hybrid_predictions.json
```

Text2midi outputs:

```powershell
.\.venv\Scripts\python.exe .\infer_midi2tag.py .\outputs\next_phase_compare\text2midi\generated_midis --model-dir .\artifacts\midi2tag --output .\outputs\midi2tag_text2midi_predictions.json
```

Each prediction contains:

- predicted tags
- confidence scores
- a short template caption
- token count

## 7. Score Prompt Alignment

Compare predicted MIDI tags against intended tags parsed from prompts:

```powershell
.\.venv\Scripts\python.exe .\score_midi2tag_alignment.py --predictions .\outputs\midi2tag_hybrid_predictions.json --prompts .\prompts_next_phase.json --output .\outputs\midi2tag_hybrid_alignment.json
```

```powershell
.\.venv\Scripts\python.exe .\score_midi2tag_alignment.py --predictions .\outputs\midi2tag_text2midi_predictions.json --prompts .\prompts_next_phase.json --output .\outputs\midi2tag_text2midi_alignment.json
```

Current local results:

```text
Hybrid mean_alignment: 0.4000
Text2midi mean_alignment: 0.2667
```

## 8. Quick Full Test Sequence

If LMD-full and `data/control_adapter_augmented/train.json` already exist:

```powershell
.\.venv\Scripts\python.exe .\prepare_midi2tag_dataset.py --metadata-json .\data\control_adapter_augmented\train.json --output-dir .\data\midi2tag --max-examples 1000 --max-seq-len 1024
.\.venv\Scripts\python.exe .\train_midi2tag.py --train .\data\midi2tag\train.json --val .\data\midi2tag\val.json --output-dir .\artifacts\midi2tag --epochs 30 --batch-size 32 --max-train-steps 1000 --embedding-dim 128 --hidden-dim 128 --max-seq-len 1024
.\.venv\Scripts\python.exe .\evaluate_midi2tag.py --model-dir .\artifacts\midi2tag --data .\data\midi2tag\test.json --output .\outputs\midi2tag_eval.json
.\.venv\Scripts\python.exe .\infer_midi2tag.py .\outputs\next_phase_compare\hybrid\generated --model-dir .\artifacts\midi2tag --output .\outputs\midi2tag_hybrid_predictions.json
.\.venv\Scripts\python.exe .\score_midi2tag_alignment.py --predictions .\outputs\midi2tag_hybrid_predictions.json --prompts .\prompts_next_phase.json --output .\outputs\midi2tag_hybrid_alignment.json
```

## Notes

- This is a small proof-of-concept classifier, not a SOTA MIDI captioning model.
- The dataset is intentionally limited to single-track MIDI to keep training fast.
- Many MidiCaps/LMD files are multi-track, so they are skipped when `--single-track-only` is enabled.
- The alignment score is only as reliable as the simple prompt parser and weak MidiCaps labels.
