# CLaMP 3 Evaluation \& RAG Pipeline — Rabbi's Branch

## Overview

This branch contains the quantitative evaluation framework for the NLP-to-MIDI pipeline,
including CLaMP 3 text alignment scoring, FAISS-based RAG retrieval, and the three-condition
ablation study (direct controls vs. prompt-to-DSL vs. RAG-augmented).

## Key Findings

|Metric|Value|
|-|-|
|Text alignment delta (C2 − C1)|−0.004|
|Cross-condition MIDI-MIDI similarity|0.922|
|RAG prompts changed|1 / 25|
|Bottleneck|DSL granularity + MIDI-GPT semantic range|

## Setup

### 1\. Clone CLaMP 3 (separate from this repo)

```bash
git clone https://github.com/sanderwood/clamp3.git
cd clamp3
conda create -n clamp3 python=3.10 -y
conda activate clamp3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install faiss-cpu wandb matplotlib openpyxl huggingface\_hub\[hf\_xet]
```

### 2\. Configure CLaMP 3 for symbolic music

Edit `clamp3/code/config.py`, line 66:

```python
CLAMP3\_VERSION = "c2"   # NOT "saas"
```

Download C2 weights (auto-downloads on first run, or manually):

```bash
cd code/
# Weights file will be downloaded automatically from HuggingFace
cd ..
```

### 3\. Copy evaluation scripts into clamp3/

Copy all `.py` files and `prompts\_crafted.json` from `rabbi/` into the `clamp3/` directory.
Also copy `rabbi/faiss\_index/` into `clamp3/faiss\_index/`.

### 4\. Download MIDICaps subset

```bash
python eval\_pipeline.py --download --subset\_size 200
python patch\_captions.py --train\_json midicaps\_subset/train.json --metadata faiss\_index/midicaps\_metadata.json
```

## Scripts

|Script|Purpose|Usage|
|-|-|-|
|`eval\_pipeline.py`|Main evaluation pipeline + wandb logging|`python eval\_pipeline.py --evaluate --generated\_dir <dir>`|
|`run\_evaluation.py`|Two-condition comparison (C1 vs C2)|`python run\_evaluation.py --outputs\_dir <dir> --prompts prompts\_crafted.json`|
|`build\_faiss\_index.py`|Build FAISS index from MIDICaps|`python build\_faiss\_index.py --midi\_dir midicaps\_subset/midi`|
|`rag\_module.py`|RAG retrieval + weighted voting (C3)|`python rag\_module.py --prompts prompts\_crafted.json --clamp3\_root .`|
|`patch\_captions.py`|Fix empty captions in FAISS metadata|`python patch\_captions.py --train\_json <path> --metadata <path>`|
|`batch\_generate.py`|Generate MIDI for all prompts × conditions|Place in midi-musicians repo, run with midigpt env|

## Reproducing Results

```bash
# Step 1: Quick validation — verify CLaMP 3 embeds correctly
python clamp3\_search.py test\_midi/happy\_full.mid test\_texts/ --top\_k 2

# Step 2: Build FAISS index
python build\_faiss\_index.py --midi\_dir midicaps\_subset/midi --subset\_meta midicaps\_subset/subset\_meta.json

# Step 3: Run two-condition evaluation (requires generated MIDI files)
python run\_evaluation.py --outputs\_dir midicaps\_subset/midi/outputs --prompts prompts\_crafted.json

# Step 4: Run RAG analysis
python rag\_module.py --prompts prompts\_crafted.json --clamp3\_root . --top\_k 3

# Step 5: Test retrieval quality
python build\_faiss\_index.py --test\_only --batch\_test --prompts prompts\_crafted.json
```

## Prompt Benchmark

`prompts\_crafted.json` contains 25 prompts matched to the DSL vocabulary from
`prompt\_to\_controls.py` (mike branch). Each prompt has:

* `text`: natural language prompt with parse\_prompt() trigger words
* `controls`: ground-truth DSL values for Condition 1

Categories: p01-p06 (unambiguous), p07-p10 (multi-attribute),
p11-p15 (ambiguous), p16-p20 (edge cases), p21-p25 (non-piano instruments).

## Results Files

* `results/evaluation\_results.json` — Raw CLaMP 3 similarity scores
* `results/condition3\_rag\_controls.json` — RAG-augmented controls for all 25 prompts
* `results/retrieval\_results.json` — FAISS retrieval results with captions
* `results/full\_evaluation\_report.xlsx` — Formatted tables for slides/poster
* `results/comparison\_heatmap.png` — Visual comparison of conditions

## Dependencies

* Python 3.10 (CLaMP 3 environment)
* PyTorch, transformers, faiss-cpu, wandb, matplotlib, openpyxl
* CLaMP 3 repo with C2 weights
* MIDICaps subset (downloaded via eval\_pipeline.py)

