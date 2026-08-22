"""
Entry point for Baseline B: Existing Unified S2ST (SeamlessM4T)
This addresses Stage 1 - Baseline B from the roadmap document.

USAGE
-----
1. Ensure your Colab has the Seamless Communication library installed:
   !pip install git+https://github.com/facebookresearch/seamless_communication.git

2. Run:  python run_baseline_b.py

Outputs land in results/ with a "_seamless" suffix.
"""
import glob
import os
import torch
import torchaudio

# Note: this requires the seamless_communication package
try:
    from seamless_communication.inference import Translator
except ImportError:
    print("Please install seamless_communication first:")
    print("pip install git+https://github.com/facebookresearch/seamless_communication.git")
    exit(1)

import config

TARGET_LANGS = ["hin", "tel", "mar"] # Seamless uses 3-letter codes

def parse_filename(path: str):
    """speakerA_hi_neutral.wav -> ("speakerA", "hi", "neutral")"""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None, None, None
    speaker = parts[0]
    lang = parts[1]
    tag = "_".join(parts[2:]) if len(parts) > 2 else "untagged"
    return speaker, lang, tag

def main():
    audio_files = sorted(glob.glob(os.path.join(config.TEST_AUDIO_DIR, "*.wav")))
    if not audio_files:
        print(f"No .wav files found in {config.TEST_AUDIO_DIR}/.")
        return

    print("Loading SeamlessM4T v2 Large model...")
    # Initialize a Translator object with a multitask model
    translator = Translator(
        model_name_or_card="seamlessM4T_v2_large",
        vocoder_name_or_card="vocoder_v2",
        device=torch.device(config.DEVICE),
        dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    total_runs = 0

    for audio_path in audio_files:
        speaker, src_lang, tag = parse_filename(audio_path)
        if src_lang not in config.LANGUAGES:
            continue
            
        src_lang_seamless = config.LANGUAGES[src_lang].get("flores", "")[:3] # mapping to 3-letter

        for tgt_lang in TARGET_LANGS:
            if tgt_lang == src_lang_seamless:
                continue

            run_id = f"{speaker}_{src_lang}_{tag}__to_{tgt_lang}_seamless"
            out_path = os.path.join(config.RESULTS_DIR, f"{run_id}.wav")
            print(f"Translating {audio_path} -> {tgt_lang} (Seamless)")
            
            try:
                # S2ST prediction
                out_texts, out_audios = translator.predict(
                    input=audio_path,
                    task_str="S2ST",
                    tgt_lang=tgt_lang,
                    src_lang=src_lang_seamless,
                )
                
                # Save audio
                torchaudio.save(
                    out_path,
                    out_audios[0].audio_wavs[0][0].cpu(),
                    sample_rate=out_audios[0].sample_rate,
                )
                print(f"Saved {out_path}")
                total_runs += 1
            except Exception as e:
                print(f"[ERROR] Seamless translation failed for {run_id}: {e}")

    print(f"Done. {total_runs} Seamless runs completed.")

if __name__ == "__main__":
    main()
