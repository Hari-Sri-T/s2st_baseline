"""
TTS wrapper around AI4Bharat's IndicF5 (zero-shot voice cloning, Indic-tuned).

NOTE ON WEIGHT FORMAT:
  The downloaded model.safetensors has keys like `ema_model._orig_mod.transformer.*`.
  But f5_tts's load_checkpoint expects plain `transformer.*` keys.
  setup.sh runs fix_weights.py to produce the corrected model.safetensors.
  This class assumes setup.sh has already been run.
"""
import numpy as np
import os
import soundfile as sf
import torch
from transformers import AutoModel


class IndicF5TTS:
    def __init__(self, checkpoint: str = "ai4bharat/IndicF5", device: str = "cuda"):
        self.device = device

        # low_cpu_mem_usage=False is required: IndicF5's custom __init__ does real
        # computation (building/compiling the vocoder) during __init__, not just
        # declaring params, which breaks under meta-device init.
        self.model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, low_cpu_mem_usage=False
        ).to(device)

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        output_path: str,
        sample_rate: int = 24000,
    ) -> str:
        """
        Zero-shot voice cloning: synthesize `text` in the voice from `ref_audio_path`.

        Args:
            text: target-language text to speak
            ref_audio_path: short reference clip of the ORIGINAL speaker
            ref_text: transcript of the reference clip (required by IndicF5)
            output_path: where to write the synthesized .wav
        Returns:
            output_path
        """
        with torch.no_grad():
            audio = self.model(
                text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
            )

        audio = np.asarray(audio, dtype=np.float32)
        # Normalize if the model returns int16-range floats
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        sf.write(output_path, audio, samplerate=sample_rate)
        return output_path
