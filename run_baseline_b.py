"""
Entry point for Baseline B: Existing Unified S2ST (SeamlessM4T)
This addresses Stage 1 - Baseline B from the roadmap document.

USAGE
-----
1. Ensure your environment has transformers and sentencepiece:
   pip install transformers sentencepiece

2. Run:  python run_baseline_b.py

Outputs land in results/ with a "_seamless" suffix.
"""
import glob
import os
import time
import torch
import torchaudio

try:
    from transformers import AutoProcessor, SeamlessM4Tv2Model
except ImportError:
    print("Please install transformers first:")
    print("pip install transformers sentencepiece")
    exit(1)

import config
from utils.logging_io import append_result

TARGET_LANGS = {"hi": "hin", "te": "tel", "mr": "mar"} # Seamless uses 3-letter codes

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

    print("Loading SeamlessM4T v2 Large model via Transformers...")
    
    device = torch.device(config.DEVICE)
    dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        torch_dtype=dtype
    ).to(device)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    total_runs = 0

    for audio_path in audio_files:
        speaker, src_lang, tag = parse_filename(audio_path)
        if src_lang not in config.LANGUAGES:
            continue
            
        # Load and resample input audio to 16kHz
        audio, orig_freq = torchaudio.load(audio_path)
        if orig_freq != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Processor expects a 1D array
        audio_array = audio.squeeze().numpy()

        for tgt_lang, tgt_lang_seamless in TARGET_LANGS.items():
            if tgt_lang == src_lang:
                continue

            run_id = f"{speaker}_{src_lang}_{tag}__to_{tgt_lang}_seamless"
            out_path = os.path.join(config.RESULTS_DIR, f"{run_id}__output.wav")
            print(f"Translating {audio_path} -> {tgt_lang_seamless} (Seamless)")
            
            try:
                started_at = time.perf_counter()
                # S2ST prediction
                audio_inputs = processor(audios=audio_array, return_tensors="pt", sampling_rate=16000).to(device)
                
                # Generate audio
                with torch.no_grad():
                    # Generate speech
                    audio_array_from_audio = model.generate(
                        **audio_inputs,
                        tgt_lang=tgt_lang_seamless,
                        return_intermediate_token_ids=False
                    )[0].cpu().numpy().squeeze()
                    
                    # Generate text
                    text_out = model.generate(
                        **audio_inputs,
                        tgt_lang=tgt_lang_seamless,
                        return_intermediate_token_ids=False,
                        generate_speech=False
                    )
                    translated_text = processor.decode(text_out[0].tolist()[0], skip_special_tokens=True)
                    
                total_seconds = time.perf_counter() - started_at
                
                # Save audio (Seamless generates audio at 16kHz)
                torchaudio.save(
                    out_path,
                    torch.from_numpy(audio_array_from_audio).unsqueeze(0),
                    sample_rate=16000,
                )
                append_result(config.RESULTS_DIR, {
                    "run_id": run_id,
                    "source_audio": audio_path,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "translated_text": translated_text,
                    "tts_backend": "seamless-m4t-v2-large",
                    "output_audio": out_path,
                    "reference_clip": None,
                    "latency": {
                        "total_seconds": round(total_seconds, 3),
                    },
                })
                print(f"Saved {out_path}")
                total_runs += 1
            except Exception as e:
                print(f"[ERROR] Seamless translation failed for {run_id}: {e}")

    print(f"Done. {total_runs} Seamless runs completed.")

if __name__ == "__main__":
    main()
