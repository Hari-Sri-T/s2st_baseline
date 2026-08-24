"""
Small audio helpers. Kept dependency-light (soundfile + numpy only) since
this runs once per pipeline call, not in a hot loop.
"""
import numpy as np
import soundfile as sf


def extract_reference_clip(audio_path: str, out_path: str, max_seconds: float = 15.0) -> str:
    """
    F5-TTS / IndicF5 both need a short reference clip of the source speaker
    to clone the voice from. They also need the exact transcript of this clip!
    Since test recordings are 3-10 seconds, we avoid truncating them so that the
    full ASR transcript correctly aligns with the full reference audio.

    Args:
        audio_path: full source recording
        out_path: where to write the reference clip
        max_seconds: warn and truncate only if the clip exceeds this length
    Returns:
        out_path
    """
    audio, sr = sf.read(audio_path)
    max_samples = int(max_seconds * sr)
    if len(audio) > max_samples:
        print(f"[WARNING] {audio_path} is longer than {max_seconds}s! Truncating, which may misalign with the full transcript.")
        clip = audio[:max_samples]
    else:
        clip = audio
    sf.write(out_path, clip, samplerate=sr)
    return out_path


def get_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate
