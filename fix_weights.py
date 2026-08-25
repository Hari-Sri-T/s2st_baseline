"""
fix_weights.py — Repair the IndicF5 model.safetensors for correct loading.

The HuggingFace checkpoint (ai4bharat/IndicF5) was saved from a torch.compile()d
model, so weight keys have two problems:
  1. They are prefixed with `ema_model.` (the wrapper module name)
  2. They contain `_orig_mod.` (torch.compile artifact)
  3. `vocoder.*` weights are included but are loaded separately — including them
     causes a key mismatch crash.

The `indicf5_model.py` loads the safetensors file directly into a bare DiT (CFM)
model using f5_tts's `load_checkpoint`. That model expects plain keys like:
    transformer.time_embed.*
    transformer.text_embed.*
    ...

So we need to:
  - Strip `ema_model.` prefix
  - Strip `_orig_mod.` infix
  - Drop all `vocoder.*` keys

Run once after `huggingface-cli download`:
    python fix_weights.py
"""
import os
from safetensors.torch import load_file, save_file


def fix_weights():
    path = "local_indicf5/model.safetensors"
    fixed_path = "local_indicf5/model_fixed.safetensors"

    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run setup.sh first.")
        return

    print(f"Loading {path}...")
    d = load_file(path)

    print(f"Original keys (first 5): {list(d.keys())[:5]}")

    new_d = {}
    skipped = 0
    for k, v in d.items():
        # Drop vocoder keys — loaded separately by load_vocoder()
        if k.startswith("vocoder."):
            skipped += 1
            continue

        # Strip `ema_model.` prefix
        if k.startswith("ema_model."):
            k = k[len("ema_model."):]

        # Strip `_orig_mod.` infix (torch.compile artifact)
        k = k.replace("_orig_mod.", "")

        new_d[k] = v

    print(f"Kept {len(new_d)} keys, skipped {skipped} vocoder keys.")
    print(f"Fixed keys (first 5): {list(new_d.keys())[:5]}")

    save_file(new_d, fixed_path, metadata={"format": "pt"})
    print(f"Saved fixed weights to: {fixed_path}")
    return fixed_path


if __name__ == "__main__":
    fix_weights()
