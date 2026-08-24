"""
Evaluates the intelligibility of the synthesized audio by computing 
Character Error Rate (CER) and Word Error Rate (WER) using Whisper.
"""
import os
import json
import jiwer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from asr.whisper_asr import WhisperASR

def resolve_project_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else config.project_path(path)

def main():
    metadata_path = os.path.join(config.RESULTS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No metadata found at {metadata_path}.")
        return
        
    with open(metadata_path, 'r') as f:
        runs = json.load(f)
        
    print(f"Loading Whisper ASR ({config.WHISPER_MODEL_SIZE}) for Intelligibility Testing...")
    asr = WhisperASR(model_size=config.WHISPER_MODEL_SIZE, device=config.DEVICE)
    
    results = {}
    
    for run in runs:
        run_id = run.get("run_id")
        tgt_audio = resolve_project_path(run.get("output_audio"))
        ground_truth_text = run.get("translated_text")
        target_lang = run.get("target_lang")
        
        if not tgt_audio or not os.path.exists(tgt_audio):
            print(f"Missing generated audio for run {run_id}")
            continue
        if not ground_truth_text:
            print(f"Skipping {run_id}: no translated_text recorded for CER/WER reference")
            continue
            
        print(f"\nEvaluating Intelligibility: {run_id}")
        
        # Determine language hint for Whisper
        whisper_hint = config.LANGUAGES.get(target_lang, {}).get("whisper", None)
        
        # Transcribe the GENERATED audio
        try:
            asr_result = asr.transcribe(tgt_audio, language_hint=whisper_hint)
            transcribed_text = asr_result["text"]
        except Exception as e:
            print(f"  [ERROR] Whisper failed to transcribe: {e}")
            transcribed_text = ""
            
        # Calculate Error Rates
        if not ground_truth_text.strip():
            print("  [WARNING] Ground truth is empty!")
            cer, wer = 1.0, 1.0
        elif not transcribed_text.strip():
            print("  [WARNING] Audio contains no speech (100% error)!")
            cer, wer = 1.0, 1.0
        else:
            try:
                cer = jiwer.cer(ground_truth_text, transcribed_text)
                wer = jiwer.wer(ground_truth_text, transcribed_text)
            except Exception as e:
                cer, wer = 1.0, 1.0
            
        results[run_id] = {
            "ground_truth": ground_truth_text,
            "transcription": transcribed_text,
            "cer": round(cer, 3),
            "wer": round(wer, 3)
        }
        
        print(f"  Ground Truth : {ground_truth_text}")
        print(f"  Transcription: {transcribed_text}")
        print(f"  --> CER: {cer:.2%} | WER: {wer:.2%}")
        
    # Save results
    out_file = os.path.join(config.RESULTS_DIR, "intelligibility_results.json")
    with open(out_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_file}")

if __name__ == "__main__":
    main()
