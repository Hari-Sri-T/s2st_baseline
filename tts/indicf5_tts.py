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

NOTE ON BLANK AUDIO BUG: The IndicF5 checkpoint on HuggingFace was saved from
a torch.compile()d model, so its keys look like `ema_model.transformer.*`.
But when AutoModel loads it, it re-applies torch.compile() internally and expects
`ema_model._orig_mod.transformer.*`. This mismatch leaves the model randomly
initialized — producing silent/blank audio. We fix this by rewriting the keys
in the safetensors file once before loading.
"""
import numpy as np
import os
import soundfile as sf
import torch
from transformers import AutoModel


def _fix_indicf5_weights(local_dir: str):
    """
    One-time fix: rewrite checkpoint keys to add `_orig_mod.` so they match
    what the internally torch.compile()d model expects.
    """
    original_path = os.path.join(local_dir, "model.safetensors")
    backup_path = original_path + ".bak"
    fixed_path = os.path.join(local_dir, "model_fixed.safetensors")

    # Already fixed
    if os.path.exists(backup_path):
        return

    if not os.path.exists(original_path):
        return

    print("[IndicF5] Fixing weight key mismatch (one-time operation)...")
    try:
        from safetensors.torch import load_file, save_file
        d = load_file(original_path)
        new_d = {}
        for k, v in d.items():
            parts = k.split(".", 1)
            if len(parts) == 2 and not parts[1].startswith("_orig_mod."):
                new_k = f"{parts[0]}._orig_mod.{parts[1]}"
            else:
                new_k = k
            new_d[new_k] = v

        save_file(new_d, fixed_path, metadata={"format": "pt"})
        # Atomically swap: backup original, put fixed in its place
        os.rename(original_path, backup_path)
        os.rename(fixed_path, original_path)
        print("[IndicF5] Weight keys fixed. Original backed up to model.safetensors.bak")
    except Exception as e:
        print(f"[IndicF5] Warning: could not fix weights ({e}). Audio may be silent.")


class IndicF5TTS:
    def __init__(self, checkpoint: str = "ai4bharat/IndicF5", device: str = "cuda"):
        self.device = device

        # Fix weight key mismatch before loading if using a local directory
        if os.path.isdir(checkpoint):
            _fix_indicf5_weights(checkpoint)

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
