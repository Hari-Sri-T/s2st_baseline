"""
ASR wrapper around openai-whisper.
Roadmap doc, Stage 1 / Baseline A: "Start with Whisper / Whisper-family model."
"""
import whisper


class WhisperASR:
    def __init__(self, model_size: str = "medium", device: str = "cuda"):
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: str, language_hint: str = None) -> dict:
        """
        Args:
            audio_path: path to source .wav file
            language_hint: whisper language code (e.g. "hi", "te", "mr").
                            If None, Whisper will auto-detect (useful for the
                            code-mixed / language-ID test category in the roadmap).
        Returns:
            {
                "text": str,             # transcript
                "language": str,         # detected/used language code
                "segments": list,        # word/segment-level timing (useful later for prosody work)
            }
        """
        # Provide a general domain/language prompt to guide Whisper and
        # disable previous text conditioning to prevent hallucinations on short clips.
        prompt = f"Here is a sentence spoken in {language_hint}." if language_hint else "Here is a spoken sentence."
        
        result = self.model.transcribe(
            audio_path,
            language=language_hint,
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        return {
            "text": result["text"].strip(),
            "language": result.get("language", language_hint),
            "segments": result.get("segments", []),
        }
