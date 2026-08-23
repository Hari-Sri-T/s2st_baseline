"""
Stage 3 - Data Pipeline: Dataset Ingestor
Downloads a monolingual Indic dataset from Hugging Face and formats it for our pipeline.
"""
import os
import json
import soundfile as sf
from datasets import load_dataset

# We use google/fleurs as it is open (unlike Common Voice which is gated)
DATASET_NAME = "google/fleurs"
CONFIG_NAME = "hi_in"  # Hindi
SPLIT = "test"
NUM_SAMPLES = 10       # Tiny subset for testing

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "raw", "hi")

def main():
    print(f"Downloading {NUM_SAMPLES} samples from {DATASET_NAME} ({CONFIG_NAME})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use streaming to avoid downloading the entire 50GB dataset for just 10 samples
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT, streaming=True)
    
    metadata = []
    
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
            
        audio = sample["audio"]
        transcript = sample["transcription"]
        # FLEURS doesn't always have explicit speaker IDs, we'll use an index if missing
        speaker_id = f"speakerFleurs_{sample.get('client_id', i)}"
        
        # We need a clean run_id compatible with our baseline naming convention
        run_id = f"{speaker_id}_hi_fleurs{i}"
        audio_filename = f"{run_id}.wav"
        audio_path = os.path.join(OUTPUT_DIR, audio_filename)
        
        # Save audio
        sf.write(audio_path, audio["array"], audio["sampling_rate"])
        
        # Append metadata
        metadata.append({
            "run_id": run_id,
            "speaker_id": speaker_id,
            "language": "hi",
            "source_audio": f"datasets/raw/hi/{audio_filename}",
            "transcript": transcript
        })
        
        print(f"[{i+1}/{NUM_SAMPLES}] Saved {audio_filename}")
        
    # Save the mapping file
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully ingested {NUM_SAMPLES} samples to {OUTPUT_DIR}")
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    main()
