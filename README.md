# midi-musicians

Workspace for exploring text-guided symbolic music generation with two complementary paths:

1. `prompt -> LLM or learned controls -> MIDI-GPT`
2. `text -> Text2midi -> MIDI`

The current `mike` branch contains the control-learning, benchmarking, and report tooling built on top of the original direct MIDI-GPT prompt wrapper.

## What is implemented

- [prompt_to_controls.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/prompt_to_controls.py)
  - Heuristic prompt parser that maps natural-language prompts to a compact structured control schema.

- [control_adapter.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/control_adapter.py)
  - Learned caption-to-control model used to predict symbolic control variables from text.

- [hybrid_control_interface.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/hybrid_control_interface.py)
  - Hybrid layer that combines learned predictions with heuristic overrides for harder prompt attributes.

- [midigpt_bridge.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/midigpt_bridge.py)
  - Converts predicted controls into a MIDI-GPT-compatible payload.

- [run_midigpt_bridge_experiment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/run_midigpt_bridge_experiment.py)
  - Runs end-to-end MIDI-GPT generation from the bridge payload.

- [benchmark_hybrid_pipeline.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/benchmark_hybrid_pipeline.py)
  - Runs a multi-prompt benchmark and stores generation metadata for comparison.

- [midi_analyzer.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/midi_analyzer.py)
  - Extracts lightweight symbolic MIDI statistics such as note density, pitch span, estimated tempo, and max polyphony.

- [score_control_alignment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/score_control_alignment.py)
  - Measures whether requested control attributes line up with observed MIDI features.

- [text2midi_runner.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/text2midi_runner.py)
  - Runs the Text2midi comparison path using released weights from Hugging Face.

- [run_text2midi_experiment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/run_text2midi_experiment.py)
  - End-to-end Text2midi prompt generation and analysis.

- [prepare_control_dataset.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/prepare_control_dataset.py)
- [prepare_augmented_control_dataset.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/prepare_augmented_control_dataset.py)
- [train_control_adapter.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/train_control_adapter.py)
- [evaluate_control_adapter.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/evaluate_control_adapter.py)
  - Data preparation, training, and evaluation utilities for the learned control interface.

## Reproducing the project

### 1. Create a Python environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The committed dependency list is in [requirements.txt](C:/Users/micha/Desktop/School/nlp/midi-musicians/requirements.txt).

### 2. Download the external assets that are not committed

This repo intentionally does **not** store large generated assets, checkpoints, datasets, or vendored builds in Git. To reproduce the full project you will need:

- `vendor/MIDI-GPT`
  - Clone the official MIDI-GPT repository into `vendor/MIDI-GPT`
  - Source used in this project: [Metacreation-Lab/MIDI-GPT](https://github.com/Metacreation-Lab/MIDI-GPT)

- MIDI-GPT checkpoint
  - Expected default checkpoint path:
    - `vendor/MIDI-GPT/models/unzipped/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt`
  - This is required by [run_midigpt_bridge_experiment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/run_midigpt_bridge_experiment.py)

- `vendor/Text2midi`
  - Clone the Text2midi repository into `vendor/Text2midi`
  - Source used in this project: [AMAAI-Lab/Text2midi](https://github.com/AMAAI-Lab/Text2midi)
  - The model weights and tokenizer are downloaded automatically from [amaai-lab/text2midi](https://huggingface.co/amaai-lab/text2midi) when [text2midi_runner.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/text2midi_runner.py) is first run

- MidiCaps dataset access
  - Required for the learned control pipeline and dataset preparation scripts
  - These scripts use `datasets.load_dataset(...)`, so you will need the corresponding dataset package support and network access at least once

- Optional Ollama installation
  - Only needed if you want to reproduce the teammate branch path based on direct LLM-to-DSL prompting
  - The committed teammate wrapper expects a local Ollama server at `http://localhost:11434`

### External asset acquisition guide

This section gives the shortest practical path to obtaining each external dependency used by the project.

#### A. MIDI-GPT source code

Clone the official MIDI-GPT repository into the expected vendor path:

```powershell
git clone https://github.com/Metacreation-Lab/MIDI-GPT.git .\vendor\MIDI-GPT
```

Expected location after cloning:

- [vendor/MIDI-GPT](C:/Users/micha/Desktop/School/nlp/midi-musicians/vendor/MIDI-GPT)

#### B. MIDI-GPT checkpoint

The bridge runner expects the checkpoint at:

- `vendor/MIDI-GPT/models/unzipped/EXPRESSIVE_ENCODER_RES_1920_12_GIGAMIDI_CKPT_150K.pt`

How to acquire it:

1. Check the official MIDI-GPT repository README and helper scripts for the model download instructions.
2. Download the released expressive encoder checkpoint and place it under:

```text
vendor/MIDI-GPT/models/unzipped/
```

If the official repo provides a zipped model archive, extract it so that the `.pt` file ends up exactly at the path above.

#### C. Text2midi source code

Clone the Text2midi repository into the expected vendor path:

```powershell
git clone https://github.com/AMAAI-Lab/Text2midi.git .\vendor\Text2midi
```

Expected location after cloning:

- [vendor/Text2midi](C:/Users/micha/Desktop/School/nlp/midi-musicians/vendor/Text2midi)

The Python runner in this repo imports the Text2midi model code directly from that folder.

#### D. Text2midi weights and tokenizer

These are downloaded automatically by [text2midi_runner.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/text2midi_runner.py) from Hugging Face the first time you run it.

Manual source:

- [amaai-lab/text2midi](https://huggingface.co/amaai-lab/text2midi)

The script pulls:

- `pytorch_model.bin`
- `vocab_remi.pkl`
- tokenizer files for `google/flan-t5-base`

So if your machine has internet access, you usually do not need to place these manually.

#### E. MidiCaps dataset

The control-learning scripts rely on the Hugging Face `datasets` loader and expect MidiCaps to be reachable from Python.

Recommended sources:

- [MidiCaps project page](https://amaai-lab.github.io/MidiCaps/)
- [Hugging Face datasets](https://huggingface.co/datasets)

Practical acquisition path:

1. Make sure `datasets` is installed from [requirements.txt](C:/Users/micha/Desktop/School/nlp/midi-musicians/requirements.txt).
2. Run the dataset preparation script once:

```powershell
.\.venv\Scripts\python.exe .\prepare_control_dataset.py --output-dir .\data\control_adapter
```

3. If the dataset loader fails, inspect the dataset identifier used in the script and fetch the corresponding dataset manually from Hugging Face, then adapt the local path if needed.

Expected generated output locations:

- [data/control_adapter](C:/Users/micha/Desktop/School/nlp/midi-musicians/data/control_adapter)
- [data/control_adapter_augmented](C:/Users/micha/Desktop/School/nlp/midi-musicians/data/control_adapter_augmented)

#### F. Ollama for the teammate path

This is only necessary if you want to reproduce the direct `prompt -> Ollama -> DSL -> MIDI-GPT` branch path in the committed [MIDI-GPT](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT) wrapper.

Install Ollama from:

- [Ollama](https://ollama.com/)

Then pull the model expected by the wrapper:

```powershell
ollama pull qwen2.5:14b-instruct
```

And start the local Ollama service so the following endpoint is available:

- `http://localhost:11434`

#### G. Windows build tools for MIDI-GPT

To build the MIDI-GPT Python extension on Windows, install:

- Visual Studio 2022 Build Tools
- CMake
- Ninja

Suggested official sources:

- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- [CMake](https://cmake.org/download/)
- [Ninja releases](https://github.com/ninja-build/ninja/releases)

After installation, the project build helper is:

- [build_midigpt_windows.bat](C:/Users/micha/Desktop/School/nlp/midi-musicians/scripts/build_midigpt_windows.bat)

### 3. Build MIDI-GPT on Windows

This branch includes a Windows build helper:

- [build_midigpt_windows.bat](C:/Users/micha/Desktop/School/nlp/midi-musicians/scripts/build_midigpt_windows.bat)

Before running it, make sure you have:

- Visual Studio 2022 Build Tools
- CMake
- Ninja
- a working PyTorch install inside `.venv`
- the `vendor/MIDI-GPT` checkout
- the `vendor/vcpkg` dependencies expected by that build flow

Then run:

```powershell
.\scripts\build_midigpt_windows.bat
```

The bridge runner expects the resulting module under:

- `vendor/MIDI-GPT/build-ninja`

### 4. Prepare training data for the learned control adapter

Create a dataset derived from MidiCaps:

```powershell
.\.venv\Scripts\python.exe .\prepare_control_dataset.py --output-dir .\data\control_adapter
```

Or the richer augmented variant:

```powershell
.\.venv\Scripts\python.exe .\prepare_augmented_control_dataset.py --output-dir .\data\control_adapter_augmented
```

### 5. Train the learned control adapter

```powershell
.\.venv\Scripts\python.exe .\train_control_adapter.py --data-dir .\data\control_adapter --output-dir .\artifacts\control_adapter_v2
```

Evaluate it with:

```powershell
.\.venv\Scripts\python.exe .\evaluate_control_adapter.py --model-dir .\artifacts\control_adapter_v2 --data-dir .\data\control_adapter --output .\outputs\adapter_eval.json
```

### 6. Run the MIDI-GPT bridge path

Generate a payload from prompts:

```powershell
.\.venv\Scripts\python.exe .\midigpt_bridge.py --prompts .\prompts_benchmark.json --output .\outputs\midigpt_bridge_payload.json
```

Run MIDI-GPT generation:

```powershell
.\.venv\Scripts\python.exe .\run_midigpt_bridge_experiment.py --payload .\outputs\midigpt_bridge_payload.json --input-midi .\outputs\text2midi_smoke\prompt_00.mid --output-dir .\outputs\midigpt_bridge_run
```

Analyze the outputs:

```powershell
.\.venv\Scripts\python.exe .\midi_analyzer.py .\outputs\midigpt_bridge_run --output .\outputs\midi_feature_summary.json
```

### 7. Run the benchmark pipeline

```powershell
.\.venv\Scripts\python.exe .\benchmark_hybrid_pipeline.py --prompts .\prompts_benchmark.json --output-dir .\outputs\benchmark_hybrid
.\.venv\Scripts\python.exe .\score_control_alignment.py --summary .\outputs\benchmark_hybrid\benchmark_summary.json --output .\outputs\benchmark_alignment.json
```

### 8. Run the Text2midi comparison path

```powershell
.\.venv\Scripts\python.exe .\run_text2midi_experiment.py --prompts .\prompts.json --output-dir .\outputs\experiment_text2midi --max-len 200 --temperature 0.9
```

## What is intentionally not committed

The following are ignored on purpose because they are too large or are machine-specific:

- [artifacts](C:/Users/micha/Desktop/School/nlp/midi-musicians/artifacts)
- [data](C:/Users/micha/Desktop/School/nlp/midi-musicians/data)
- [outputs](C:/Users/micha/Desktop/School/nlp/midi-musicians/outputs)
- [vendor](C:/Users/micha/Desktop/School/nlp/midi-musicians/vendor)
- local checkpoints, model weights, archives, and compiled native binaries

If you want to reproduce exact experimental outputs, you will need to regenerate those folders locally.

## Report assets

The report materials live in:

- [report/midway_report.tex](C:/Users/micha/Desktop/School/nlp/midi-musicians/report/midway_report.tex)
- [report/custom.bib](C:/Users/micha/Desktop/School/nlp/midi-musicians/report/custom.bib)
- [report/path_comparison.md](C:/Users/micha/Desktop/School/nlp/midi-musicians/report/path_comparison.md)
