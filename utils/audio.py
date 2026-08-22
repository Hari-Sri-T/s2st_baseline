"""
Small audio helpers. Kept dependency-light (soundfile + numpy only) since
this runs once per pipeline call, not in a hot loop.
"""
import numpy as np
import soundfile as sf


def extract_reference_clip(audio_path: str, out_path: str, max_seconds: float = 8.0) -> str:
    """
    F5-TTS / IndicF5 both need a short reference clip of the source speaker
    to clone the voice from. Whisper's full source recording may be longer
    than ideal for this, so trim to the first `max_seconds`.

    Args:
        audio_path: full source recording
        out_path: where to write the trimmed reference clip
        max_seconds: reference clip length (5-10s is the usual sweet spot
                     for these zero-shot cloning models)
    Returns:
        out_path
    """
    audio, sr = sf.read(audio_path)
    max_samples = int(max_seconds * sr)
    clip = audio[:max_samples] if len(audio) > max_samples else audio
    sf.write(out_path, clip, samplerate=sr)
    return out_path


def get_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate
