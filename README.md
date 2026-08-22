# Week 1 Baseline — Cascaded S2ST Pipeline

Implements Baseline A from the roadmap doc:

```
MIC/AUDIO → ASR (Whisper) → IndicTrans2 → Voice/Expressive TTS (IndicF5 or F5-TTS) → TARGET SPEECH
```

Run this in **Colab with a GPU runtime** — it needs to download multi-GB model
weights and do GPU inference, neither of which works in a plain CPU sandbox.

## Setup (Colab)

```python
# Cell 1 - runtime check
!nvidia-smi   # confirm you have a GPU (Runtime > Change runtime type > GPU)

# Cell 2 - get the code (upload this folder as a zip, or push to a private repo and clone)
!unzip s2st_baseline.zip
%cd s2st_baseline

# Cell 3 - install deps
!pip install -r requirements.txt
!pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

First run will download:
- Whisper `medium` (~1.5GB)
- IndicTrans2 checkpoint(s) (~1-2GB each, only the direction(s) you actually use get loaded)
- IndicF5 or F5-TTS weights (~1-2GB)

All from Hugging Face Hub — make sure the Colab runtime has internet (it does by default).

## Add your test recordings

Drop `.wav` files into `test_audio/` using this naming convention:

```
{speaker}_{source_lang}_{tag}.wav
```

Examples:
```
speakerA_hi_neutral.wav
speakerA_te_excited.wav
speakerB_mr_neutral.wav
```

`source_lang` must be `hi`, `te`, `mr`, or `en` (edit `config.py` → `LANGUAGES` to add more).
`tag` is free text for your own bookkeeping — doesn't affect processing yet, but
name it sensibly now (`neutral`/`happy`/`sad`/etc.) since Week 2's emotion testing
will reuse this same file-naming convention.

Aim for **3-10 second clips** — that's the sweet spot for the zero-shot voice
cloning TTS models to get a clean speaker reference.

## Run

```python
!python run_baseline.py
```

This will, for every recording in `test_audio/`, translate it into every
language listed in `TARGET_LANGS` (top of `run_baseline.py`, currently
`["hi", "te", "mr"]`) and write:

- `results/<run_id>__output.wav` — the translated, voice-cloned output
- `results/<run_id>__ref.wav` — the reference clip extracted for cloning
- `results/metadata.json` — running log with transcript, translation, and
  file paths for every run (this is what Week 2's failure matrix reads from)

## Switching TTS backend

Roadmap doc says to evaluate both IndicF5 and F5-TTS. Flip one line:

```python
# config.py
TTS_BACKEND = "indicf5"   # or "f5"
```

Re-run `run_baseline.py` with each setting, keep both `results/` folders
(rename between runs, e.g. `results_indicf5/` vs `results_f5/`) so you can
A/B them side by side going into Week 2.

## What's NOT in this yet (by design — see project timeline)

- **Baseline B** (Seamless/UniSS comparison) — separate script, Week 1 Days 3-4
- **Failure testing harness** (speaker similarity, emotion, prosody metrics) — Week 2
- **Streaming/real-time** — later stage, don't build yet
- **Code-mixed input handling** — Whisper's `language_hint=None` auto-detect
  path is already wired for this in `asr/whisper_asr.py`, but nothing calls
  it that way yet; that's a Week 2 test category

## Known rough edges to expect

- **IndicF5's exact call signature** may have shifted since this was written —
  if `tts/indicf5_tts.py` throws a `TypeError` on the `self.model(...)` call,
  check the current usage snippet on the [model card](https://huggingface.co/ai4bharat/IndicF5)
  and adjust just that one call.
- **IndicTrans2 model loading** downloads whichever of the three checkpoints
  (en-indic / indic-en / indic-indic) your first translation direction needs —
  if you're translating Telugu→Hindi, that's the indic-indic checkpoint, ~2GB,
  first call will be slow.
- **Colab free-tier T4 GPU** should handle `WHISPER_MODEL_SIZE = "medium"` plus
  IndicTrans2 1B plus IndicF5 fine, but if you hit OOM, drop Whisper to `"small"`
  in `config.py` first — it's the cheapest quality trade-off of the three.
