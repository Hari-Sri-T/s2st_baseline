"""
Lightweight preflight for the S2ST baseline environment.

This avoids importing large model classes; it only checks whether the expected
packages, tools, folders, and audio inputs are present before a long Colab run.
"""
import glob
import importlib.util
import os
import shutil

import config


REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("openai-whisper", "whisper"),
    ("f5-tts", "f5_tts"),
]

OPTIONAL_EVAL_PACKAGES = [
    ("speechbrain", "speechbrain"),
    ("torchaudio", "torchaudio"),
    ("librosa", "librosa"),
    ("jiwer", "jiwer"),
]


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def print_check(ok: bool, label: str, detail: str = ""):
    status = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def main():
    print("S2ST baseline environment check")
    print(f"Project: {config.BASE_DIR}")
    print(f"Device : {config.DEVICE}")

    for label, module_name in REQUIRED_PACKAGES:
        print_check(has_module(module_name), label)

    for label, module_name in OPTIONAL_EVAL_PACKAGES:
        print_check(has_module(module_name), label)

    print_check(shutil.which("ffmpeg") is not None, "ffmpeg")
    print_check(os.path.isdir(config.TEST_AUDIO_DIR), "test_audio directory", config.TEST_AUDIO_DIR)

    audio_files = glob.glob(os.path.join(config.TEST_AUDIO_DIR, "*.wav"))
    print_check(len(audio_files) > 0, "test .wav files", f"{len(audio_files)} found")

    if config.TTS_BACKEND == "indicf5":
        print_check(
            os.path.isdir(config.INDICF5_CHECKPOINT),
            "IndicF5 local checkpoint",
            config.INDICF5_CHECKPOINT,
        )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print_check(os.path.isdir(config.RESULTS_DIR), "results directory", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
