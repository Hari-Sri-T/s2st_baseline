"""
TTS wrapper around AI4Bharat's IndicF5 (zero-shot voice cloning, Indic-tuned).
Roadmap doc, Stage 1 / Baseline A TTS candidate — preferred first choice since
it's Indic-specific rather than general multilingual.

Requires: pip install transformers accelerate soundfile

NOTE ON API STABILITY: IndicF5's exact call signature has moved before as
AI4Bharat updates the model card. If `model(...)` below throws a TypeError,
check https://huggingface.co/ai4bharat/IndicF5 for the current usage snippet
and adjust the `synthesize` method's call — the loading and file-saving logic
around it won't need to change.
"""
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


class IndicF5TTS:
    def __init__(self, checkpoint: str = "ai4bharat/IndicF5", device: str = "cuda"):
        self.device = device
        # low_cpu_mem_usage=False is required here: transformers' default
        # fast-init path builds the model on a fake "meta" device before
        # loading real weights. IndicF5's custom __init__ does real
        # computation (building/compiling the vocoder) during __init__,
        # not just declaring params, which breaks under meta-device init
        # with "Tensor on device cpu is not on the expected device meta".
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
                             (this is what carries speaker identity across languages)
            ref_text: transcript of the reference clip (required by IndicF5's
                       prompt-conditioning scheme)
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
