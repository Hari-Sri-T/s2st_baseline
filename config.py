"""
Central config for the Stage-1 cascaded baseline.
MIC/AUDIO -> ASR (Whisper) -> IndicTrans2 -> TTS (F5-TTS / IndicF5) -> TARGET SPEECH

Edit LANGUAGES to add/remove languages. Everything else keys off this dict.
"""

# ---- Languages we're starting with (per roadmap doc: Hindi, Telugu, Marathi first) ----
# FLORES-200 codes are what IndicTrans2 expects.
LANGUAGES = {
    "hi": {"name": "Hindi",   "flores": "hin_Deva", "whisper": "hi"},
    "te": {"name": "Telugu",  "flores": "tel_Telu", "whisper": "te"},
    "mr": {"name": "Marathi", "flores": "mar_Deva", "whisper": "mr"},
    "en": {"name": "English", "flores": "eng_Latn", "whisper": "en"},
}

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


# ---- ASR ----
WHISPER_MODEL_SIZE = "medium"   # "small" if you're VRAM constrained on Colab free tier

# ---- Translation (IndicTrans2 checkpoints, AI4Bharat) ----
# Pick checkpoint based on src/tgt direction at runtime (see translate/indictrans2.py)
INDICTRANS2_CHECKPOINTS = {
    "en-indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic-en": "ai4bharat/indictrans2-indic-en-1B",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-1B",
}

# ---- TTS ----
# "f5" = SWivid/F5-TTS (strong zero-shot voice cloning, general multilingual-ish)
# "indicf5" = ai4bharat/IndicF5 (Indic-tuned, prefer this for Indic-target quality first)
TTS_BACKEND = "indicf5"   # switch to "f5" to A/B compare, per roadmap doc Stage 1 TTS candidates
INDICF5_CHECKPOINT = project_path("local_indicf5")
F5TTS_CHECKPOINT = "SWivid/F5-TTS"

# ---- I/O ----
TEST_AUDIO_DIR = project_path("test_audio")     # put source recordings here, named like: speakerA_hi_neutral.wav
RESULTS_DIR = project_path("results")           # pipeline writes input/output pairs + metadata.json here
SAMPLE_RATE = 16000                # ASR expects 16k; TTS output resampled separately if needed

# ---- Device ----
try:
    import torch

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"
