"""
The Baseline A pipeline from the roadmap doc:

  MIC/AUDIO -> ASR -> Indic Translation -> Voice/Expressive TTS -> TARGET SPEECH

This module wires the three components together for a single input file.
See run_baseline.py for batch-running this over test_audio/.
"""
import os

from asr.whisper_asr import WhisperASR
from translate.indictrans2 import IndicTrans2Translator
from tts.factory import build_tts
from utils.audio import extract_reference_clip
from utils.logging_io import append_result

import config


class BaselinePipeline:
    def __init__(self):
        print(f"[pipeline] Loading Whisper ({config.WHISPER_MODEL_SIZE})...")
        self.asr = WhisperASR(model_size=config.WHISPER_MODEL_SIZE, device=config.DEVICE)

        print("[pipeline] Loading IndicTrans2...")
        self.translator = IndicTrans2Translator(
            checkpoints=config.INDICTRANS2_CHECKPOINTS, device=config.DEVICE
        )

        print(f"[pipeline] Loading TTS backend: {config.TTS_BACKEND}...")
        tts_checkpoint = (
            config.INDICF5_CHECKPOINT if config.TTS_BACKEND == "indicf5" else config.F5TTS_CHECKPOINT
        )
        self.tts = build_tts(config.TTS_BACKEND, tts_checkpoint, config.DEVICE)

    def run(
        self,
        source_audio_path: str,
        source_lang: str,
        target_lang: str,
        run_id: str,
    ) -> dict:
        """
        Args:
            source_audio_path: path to the source .wav
            source_lang: 2-letter code, key into config.LANGUAGES (e.g. "te")
            target_lang: 2-letter code, key into config.LANGUAGES (e.g. "hi")
            run_id: unique name for this run, used to name output files
                    (e.g. "speakerA_te_neutral__to_hi")
        Returns:
            dict record of the full run (also appended to results/metadata.json)
        """
        assert source_lang in config.LANGUAGES, f"Unknown source_lang '{source_lang}'"
        assert target_lang in config.LANGUAGES, f"Unknown target_lang '{target_lang}'"

        out_dir = config.RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)

        # 1. ASR
        whisper_hint = config.LANGUAGES[source_lang]["whisper"]
        asr_result = self.asr.transcribe(source_audio_path, language_hint=whisper_hint)
        print(f"[ASR]   ({asr_result['language']}) {asr_result['text']}")

        # 2. Translation
        src_flores = config.LANGUAGES[source_lang]["flores"]
        tgt_flores = config.LANGUAGES[target_lang]["flores"]
        translated_text = self.translator.translate(
            asr_result["text"], src_flores=src_flores, tgt_flores=tgt_flores
        )
        print(f"[MT]    -> {translated_text}")

        # 3. Reference clip for voice cloning (carries speaker identity across languages)
        ref_clip_path = os.path.join(out_dir, f"{run_id}__ref.wav")
        extract_reference_clip(source_audio_path, ref_clip_path)

        # 4. TTS (voice-cloned into target language)
        output_audio_path = os.path.join(out_dir, f"{run_id}__output.wav")
        self.tts.synthesize(
            text=translated_text,
            ref_audio_path=ref_clip_path,
            ref_text=asr_result["text"],
            output_path=output_audio_path,
        )
        print(f"[TTS]   wrote {output_audio_path}")

        record = {
            "run_id": run_id,
            "source_audio": source_audio_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "transcript": asr_result["text"],
            "detected_language": asr_result["language"],
            "translated_text": translated_text,
            "tts_backend": config.TTS_BACKEND,
            "output_audio": output_audio_path,
            "reference_clip": ref_clip_path,
        }
        append_result(out_dir, record)
        return record
