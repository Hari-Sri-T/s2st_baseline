"""
Evaluates emotion preservation between source audio and generated target audio.
Part of the Failure Testing Harness (Stage 2).
"""
import os
import json
import torch
import torchaudio
import numpy as np
from transformers import pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def resolve_project_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else config.project_path(path)

def main():
    print("Loading Speech Emotion Recognition (SER) model...")
    # Using SUPERB Emotion Recognition model (neutral, happy, angry, sad)
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=0 if config.DEVICE=="cuda" else -1)
    
    metadata_path = os.path.join(config.RESULTS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No metadata found at {metadata_path}. Have you run the baseline yet?")
        return
        
    with open(metadata_path, 'r') as f:
        runs = json.load(f)
        
    results = {}
    
    for run in runs:
        run_id = run.get("run_id")
        src_audio = resolve_project_path(run.get("source_audio"))
        tgt_audio = resolve_project_path(
            run.get("output_audio") or os.path.join("results", f"{run_id}__output.wav")
        )
        
        if not src_audio or not tgt_audio or not os.path.exists(src_audio) or not os.path.exists(tgt_audio):
            print(f"Missing audio files for run {run_id}")
            continue
            
        print(f"Evaluating Emotion: {run_id}")
        
        # Load audio
        sig1, fs1 = torchaudio.load(src_audio)
        sig2, fs2 = torchaudio.load(tgt_audio)
        
        # The model expects 16kHz
        if fs1 != 16000:
            sig1 = torchaudio.functional.resample(sig1, fs1, 16000)
        if fs2 != 16000:
            sig2 = torchaudio.functional.resample(sig2, fs2, 16000)
            
        # Get predictions
        src_preds = classifier(sig1.squeeze().numpy())
        tgt_preds = classifier(sig2.squeeze().numpy())
        
        # Get top predicted emotion
        src_top_emotion = src_preds[0]['label']
        src_top_score = src_preds[0]['score']
        
        tgt_top_emotion = tgt_preds[0]['label']
        tgt_top_score = tgt_preds[0]['score']
        
        match = (src_top_emotion == tgt_top_emotion)
        
        results[run_id] = {
            "source_emotion": src_top_emotion,
            "source_score": round(src_top_score, 4),
            "target_emotion": tgt_top_emotion,
            "target_score": round(tgt_top_score, 4),
            "match": match
        }
        
        print(f"  Source: {src_top_emotion} ({src_top_score:.2f}) -> Target: {tgt_top_emotion} ({tgt_top_score:.2f}) | Match: {match}")
        
    # Save results
    out_file = os.path.join(config.RESULTS_DIR, "emotion_preservation_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
