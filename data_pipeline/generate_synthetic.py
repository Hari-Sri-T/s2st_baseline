"""
Stage 3 - Data Pipeline: Synthetic Parallel Generator
Implements Section 16 of the roadmap:
Indic source speech ➡️ Whisper ➡️ IndicTrans2 ➡️ IndicF5 ➡️ Synthetic Target Speech
"""
import os
import json

import sys
# Make config accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.baseline_pipeline import BaselinePipeline

INPUT_METADATA = config.project_path("datasets", "raw", "hi", "metadata_annotated.json")
OUTPUT_DIR = config.project_path("datasets", "synthetic")

# For testing, we'll just generate Telugu
TARGET_LANGS = ["te"]

def main():
    if not os.path.exists(INPUT_METADATA):
        print(f"Error: Could not find {INPUT_METADATA}. Run annotate_expressive.py first.")
        return
        
    print("Loading Baseline Pipeline for Synthetic Generation...")
    pipeline = BaselinePipeline(results_dir=OUTPUT_DIR)
    
    with open(INPUT_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    synthetic_metadata = []
    
    print(f"Generating synthetic parallel audio for {len(metadata)} samples...")
    
    for i, item in enumerate(metadata):
        audio_rel = item["source_audio"]
        audio_abs = config.project_path(audio_rel)
        src_lang = item["language"]
        run_id_base = item["run_id"]
        
        for tgt_lang in TARGET_LANGS:
            if tgt_lang == src_lang:
                continue
                
            run_id = f"{run_id_base}__to_{tgt_lang}"
            
            print(f"[{i+1}/{len(metadata)}] Synthesizing {run_id_base} -> {tgt_lang}")
            
            try:
                result = pipeline.run(
                    source_audio_path=audio_abs,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    run_id=run_id,
                )
                
                # Create the parallel paired record
                synthetic_metadata.append({
                    "run_id": run_id,
                    "speaker_id": item["speaker_id"],
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "source_audio": audio_rel,
                    "target_audio": result["output_audio"],
                    "source_emotion": item["emotion"],
                    "source_transcript": item["transcript"],
                    "target_transcript": result["translated_text"],
                    "latency": result.get("latency", {}),
                })
            except Exception as e:
                print(f"[ERROR] Failed to synthesize {run_id}: {e}")
                
    # Save the synthetic mapping file
    out_meta = os.path.join(OUTPUT_DIR, "synthetic_metadata.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(synthetic_metadata, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated synthetic dataset to {OUTPUT_DIR}")
    print(f"Parallel mapping saved to {out_meta}")

if __name__ == "__main__":
    main()
