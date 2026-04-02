## Path Comparison: Teammate MIDI-GPT Wrapper vs Local Hybrid Pipeline

### Short answer

Yes, a comparison is possible in principle, but the teammate path as committed in the main repo is **not self-contained enough to run by itself yet**.

### What is now in the repo

Tracked teammate files live under:

- [MIDI-GPT/GPU_cmd.sh](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/GPU_cmd.sh)
- [MIDI-GPT/python_scripts_for_testing/LLM_inference.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/LLM_inference.py)
- [MIDI-GPT/python_scripts_for_testing/build_params.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/build_params.py)
- [MIDI-GPT/python_scripts_for_testing/post_processing.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/post_processing.py)
- [MIDI-GPT/python_scripts_for_testing/resolve_conflict.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/resolve_conflict.py)
- [MIDI-GPT/python_scripts_for_testing/pythoninferencetest.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/pythoninferencetest.py)
- [MIDI-GPT/python_scripts_for_testing/utils.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/utils.py)

### What the teammate path does

This path is a direct `prompt -> LLM -> DSL -> MIDI-GPT` wrapper:

1. A natural-language prompt is sent to Ollama.
2. The LLM is constrained to return a small JSON DSL.
3. The JSON is sanitized and normalized.
4. The controls are converted into a MIDI-GPT status structure.
5. MIDI-GPT generates output from a seed MIDI file.

### What the local path does

The local experimental path built in this workspace is a `prompt -> learned controls -> hybrid correction -> MIDI-GPT` pipeline:

- [control_adapter.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/control_adapter.py)
- [hybrid_control_interface.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/hybrid_control_interface.py)
- [midigpt_bridge.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/midigpt_bridge.py)
- [run_midigpt_bridge_experiment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/run_midigpt_bridge_experiment.py)
- [benchmark_hybrid_pipeline.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/benchmark_hybrid_pipeline.py)
- [score_control_alignment.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/score_control_alignment.py)

This path adds:

- learned caption-to-control supervision from MidiCaps
- a hybrid correction layer
- multitrack control application
- benchmark prompts
- automatic feature extraction and alignment scoring

### Direct comparison dimensions

These two paths are comparable on the following axes:

| Dimension | Teammate path | Local path |
| --- | --- | --- |
| Prompt interface | LLM-generated DSL | Learned + hybrid controls |
| Conditioning source | Free-form LLM output | Trained adapter on MidiCaps-derived labels |
| Control cleaning | Rule-based sanitization | Rule-based plus learned predictions |
| MIDI-GPT integration | Single wrapper script | End-to-end bridge and benchmark runner |
| Track coverage | Primarily first track | Multitrack runtime control |
| Evaluation | Manual inspection / qualitative | Automatic MIDI feature analysis and alignment metrics |

### Why a runtime comparison is not immediate yet

The teammate path as committed in the main repo is only the wrapper layer. It does **not** currently include:

- a built `python_lib/midigpt` package inside the tracked `MIDI-GPT` folder
- MIDI-GPT models/checkpoints inside that tracked folder
- a local Ollama installation in this environment

Current practical blockers:

1. [MIDI-GPT/python_scripts_for_testing/LLM_inference.py](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/python_scripts_for_testing/LLM_inference.py) expects a local `midigpt` Python library on `sys.path`.
2. [MIDI-GPT/GPU_cmd.sh](C:/Users/micha/Desktop/School/nlp/midi-musicians/MIDI-GPT/GPU_cmd.sh) assumes a Linux/Ubuntu CUDA setup and a full MIDI-GPT checkout.
3. `ollama` is not currently installed in this Windows environment.

### Why comparison is still feasible

A comparison becomes feasible with relatively small glue work because this workspace already has:

- a built local MIDI-GPT backend under [vendor/MIDI-GPT](C:/Users/micha/Desktop/School/nlp/midi-musicians/vendor/MIDI-GPT)
- local checkpoints already used by the hybrid pipeline
- prompt sets and evaluation scripts already implemented

So the teammate path can be evaluated by:

1. Redirecting its scripts to the existing local MIDI-GPT build.
2. Replacing the missing Ollama dependency with either:
   - a local Ollama install, or
   - a drop-in prompt-to-DSL stub for controlled testing.
3. Running the same prompt benchmark and the same feature extraction/evaluation code.

### Best report framing

For the midway report, the comparison can be presented as:

- **Path A:** direct LLM-to-DSL control of MIDI-GPT
- **Path B:** learned/hybrid control prediction before MIDI-GPT

This is a meaningful comparison because the core question is whether semantic control should come mainly from:

- a powerful generative language model producing structured controls, or
- a trained control predictor tuned to symbolic music supervision.

### Recommendation

The cleanest comparison to run next is:

1. Use the same 12 benchmark prompts.
2. Feed them through the teammate DSL wrapper.
3. Feed them through the local hybrid pipeline.
4. Score both with the same feature extractor and control-alignment metrics.

That would give a fair apples-to-apples comparison between:

- a rule-and-LLM driven control interface
- a learned control interface
