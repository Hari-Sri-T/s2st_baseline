"""
TTS wrapper around SWivid/F5-TTS.
Roadmap doc, Stage 1 / Baseline A TTS candidate — "especially useful as a
voice-cloning/expressive-TTS baseline." Use this as an A/B comparison
against IndicF5 to see which handles your target languages/speakers better.

Requires: pip install f5-tts
"""
from f5_tts.api import F5TTS


class F5TTSWrapper:
    def __init__(self, checkpoint: str = "SWivid/F5-TTS", device: str = "cuda"):
        self.model = F5TTS(model="F5TTS_v1_Base", device=device)
        # checkpoint arg kept for interface parity with IndicF5TTS / config.py;
        # f5_tts's API pulls the default checkpoint automatically.

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
        Same interface as IndicF5TTS.synthesize so pipeline.py can swap backends
        without changing calling code.
        """
        wav, sr, _ = self.model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=text,
            file_wave=output_path,
        )
        return output_path
