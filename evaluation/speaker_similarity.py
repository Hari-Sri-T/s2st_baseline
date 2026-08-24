"""
Evaluates speaker similarity between source audio and generated target audio.
Part of the Failure Testing Harness (Stage 2).
"""
import os
import json
import torch
import torchaudio

# We will use SpeechBrain for speaker embeddings.
# !pip install speechbrain
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    print("Please install speechbrain first:")
    print("pip install speechbrain")
    exit(1)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def resolve_project_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else config.project_path(path)

def get_cosine_similarity(emb1, emb2):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(emb1, emb2).item()

def main():
    print("Loading speaker embedding model...")
    # Load model from HuggingFace
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
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
            
        print(f"Evaluating: {run_id}")
        
        # Load audio and compute embeddings
        sig1, fs1 = torchaudio.load(src_audio)
        sig2, fs2 = torchaudio.load(tgt_audio)
        
        # SpeechBrain model expects 16kHz
        if fs1 != 16000:
            sig1 = torchaudio.functional.resample(sig1, fs1, 16000)
        if fs2 != 16000:
            sig2 = torchaudio.functional.resample(sig2, fs2, 16000)
            
        emb1 = classifier.encode_batch(sig1)
        emb2 = classifier.encode_batch(sig2)
        
        sim = get_cosine_similarity(emb1.squeeze(), emb2.squeeze())
        results[run_id] = sim
        print(f"  Similarity: {sim:.4f}")
        
    # Save results
    out_file = os.path.join(config.RESULTS_DIR, "speaker_similarity_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
