"""
Evaluates prosody preservation (pitch and duration) between source audio and generated target audio.
Part of the Failure Testing Harness (Stage 2).
"""
import os
import json
import librosa
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def resolve_project_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else config.project_path(path)

def extract_prosody(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract pitch using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # Filter out unvoiced frames (NaNs)
    f0_voiced = f0[~np.isnan(f0)]
    
    if len(f0_voiced) > 0:
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
    else:
        f0_mean = 0.0
        f0_std = 0.0
        
    return {
        "duration": float(duration),
        "f0_mean": float(f0_mean),
        "f0_std": float(f0_std)
    }

def main():
    print("Evaluating Prosody (Pitch & Duration)...")
    
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
            
        print(f"Analyzing Prosody: {run_id}")
        
        src_metrics = extract_prosody(src_audio)
        tgt_metrics = extract_prosody(tgt_audio)
        
        duration_ratio = tgt_metrics["duration"] / src_metrics["duration"] if src_metrics["duration"] > 0 else 0
        
        # Absolute percent difference in F0 Mean
        f0_diff_percent = abs(tgt_metrics["f0_mean"] - src_metrics["f0_mean"]) / src_metrics["f0_mean"] * 100 if src_metrics["f0_mean"] > 0 else 0
        
        results[run_id] = {
            "source": src_metrics,
            "target": tgt_metrics,
            "duration_ratio": round(duration_ratio, 2),
            "f0_mean_diff_percent": round(f0_diff_percent, 2)
        }
        
        print(f"  Duration Ratio (Target/Source): {duration_ratio:.2f}x")
        print(f"  F0 Mean Diff: {f0_diff_percent:.1f}% | Src F0 Std: {src_metrics['f0_std']:.1f} Hz -> Tgt F0 Std: {tgt_metrics['f0_std']:.1f} Hz")
        
    # Save results
    out_file = os.path.join(config.RESULTS_DIR, "prosody_preservation_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
