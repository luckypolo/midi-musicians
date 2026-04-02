# midi-musicians

Workspace for exploring text-guided symbolic music generation with two complementary paths:

1. `prompt -> structured controls -> symbolic generator`
2. `text -> Text2midi -> MIDI`

## What is implemented

- `text2midi_runner.py`
  - Downloads the released `amaai-lab/text2midi` model artifacts from Hugging Face.
  - Generates MIDI files from prompt JSON files.
  - Writes a generation manifest beside the outputs.

- `prompt_to_controls.py`
  - Converts natural-language prompts into a lightweight control schema inspired by the MIDI-GPT control surface.
  - Extracts fields like mood, genre, density, register, key, and tempo hints.

- `midi_analyzer.py`
  - Extracts lightweight MIDI features such as note density, pitch span, estimated tempo, and max polyphony.

- `quick_check_caption_quality.py`
  - Runs a small heuristic quality check on prompt or caption files.

- `run_text2midi_experiment.py`
  - End-to-end experiment runner:
    - prompt parsing
    - Text2midi generation
    - MIDI feature analysis
    - consolidated JSON report

## Environment

This repo currently uses a local virtual environment:

```powershell
C:\Users\micha\Desktop\School\nlp\midi-musicians\.venv
```

The minimal project dependency list is stored in [requirements-project.txt](C:/Users/micha/Desktop/School/nlp/midi-musicians/requirements-project.txt).

## Example commands

Run the prompt-to-controls parser:

```powershell
& '.\.venv\Scripts\python.exe' '.\prompt_to_controls.py' --prompts '.\prompts.json' --output '.\outputs\prompt_controls.json'
```

Run a full Text2midi experiment:

```powershell
& '.\.venv\Scripts\python.exe' '.\run_text2midi_experiment.py' --prompts '.\prompts.json' --output-dir '.\outputs\experiment_text2midi' --max-len 200 --temperature 0.9
```

Analyze generated MIDI files:

```powershell
& '.\.venv\Scripts\python.exe' '.\midi_analyzer.py' '.\outputs\experiment_text2midi\generated_midis' --output '.\outputs\midi_feature_summary.json'
```

## Current limitation

The natural-language-to-controls product direction is scaffolded here, but the repo does not yet include a local MIDI-GPT codebase or inference wrapper. That means the control parser is ready, but true `prompt -> controls -> MIDI-GPT` testing still needs the MIDI-GPT project to be brought into the workspace and wired into the same experiment flow.
