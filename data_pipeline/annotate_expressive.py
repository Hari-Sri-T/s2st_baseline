"""
Stage 3 - Data Pipeline: Expressive Annotator
Reads ingested raw datasets and appends Emotion and Prosody tags to fulfill Category C (Expressive data).
"""
import os
import json
import torch
import torchaudio
import librosa
import numpy as np
from transformers import pipeline

import sys
# Make config accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

INPUT_METADATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "raw", "hi", "metadata.json")
OUTPUT_METADATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "raw", "hi", "metadata_annotated.json")

def extract_prosody(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_voiced = f0[~np.isnan(f0)]
    f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    return float(duration), f0_mean, f0_std

def main():
    if not os.path.exists(INPUT_METADATA):
        print(f"Error: Could not find {INPUT_METADATA}. Run ingest_hf_dataset.py first.")
        return
        
    print("Loading Emotion Recognition model...")
    device_id = 0 if config.DEVICE == "cuda" else -1
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device_id)
    
    with open(INPUT_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    print(f"Annotating {len(metadata)} samples with Emotion and Prosody...")
    
    annotated = []
    
    for i, item in enumerate(metadata):
        # Resolve absolute path
        audio_rel = item["source_audio"]
        audio_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", audio_rel)
        
        # 1. Prosody
        duration, f0_mean, f0_std = extract_prosody(audio_abs)
        
        # 2. Emotion
        sig, fs = torchaudio.load(audio_abs)
        if fs != 16000:
            sig = torchaudio.functional.resample(sig, fs, 16000)
            
        preds = classifier(sig.squeeze().numpy())
        emotion_label = preds[0]['label']
        
        # Update metadata
        item["duration"] = round(duration, 2)
        item["f0_mean"] = round(f0_mean, 2)
        item["f0_std"] = round(f0_std, 2)
        item["emotion"] = emotion_label
        
        annotated.append(item)
        print(f"[{i+1}/{len(metadata)}] {item['run_id']} -> Emotion: {emotion_label}, F0 Mean: {item['f0_mean']}Hz")
        
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved annotated metadata to {OUTPUT_METADATA}")

if __name__ == "__main__":
    main()
