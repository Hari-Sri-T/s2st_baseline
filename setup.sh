#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Downloading ai4bharat/IndicF5 model..."
# Download model without symlinks
huggingface-cli download ai4bharat/IndicF5 --local-dir local_indicf5 --local-dir-use-symlinks False

echo "Applying model code patches..."
# Overwrite the original huggingface model with our patched version
cp ./indicf5_model.py local_indicf5/model.py

echo "Applying safetensors weight fixes..."
# Run the python script to strip invalid prefixes and vocoder keys
python fix_weights.py

# Replace the original broken safetensors file with our fixed version
mv local_indicf5/model_fixed.safetensors local_indicf5/model.safetensors

echo "Setup complete! The model weights have been downloaded and fixed. You can now run the pipeline."
