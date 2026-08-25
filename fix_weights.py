import os
from safetensors.torch import load_file, save_file

def fix_weights():
    """
    The IndicF5 checkpoint on HuggingFace was saved from a torch.compile()d model,
    so its weight keys look like:
        ema_model.transformer.*
        vocoder.*

    But the AutoModel loads the model via torch.compile() again internally, so it
    expects keys like:
        ema_model._orig_mod.transformer.*
        vocoder._orig_mod.*

    This script rewrites the keys to match what the model actually expects.
    Run this ONCE after setup.sh, before running any inference.
    """
    path = "local_indicf5/model.safetensors"
    fixed_path = path.replace(".safetensors", "_fixed.safetensors")

    if os.path.exists(fixed_path):
        print(f"Fixed weights already exist at {fixed_path}, skipping.")
        return fixed_path

    print("Fixing IndicF5 weight keys:", path)
    d = load_file(path)
    new_d = {}
    for k, v in d.items():
        # Insert _orig_mod. after the top-level module name
        # e.g. ema_model.transformer.* -> ema_model._orig_mod.transformer.*
        # e.g. vocoder.backbone.* -> vocoder._orig_mod.backbone.*
        parts = k.split(".", 1)
        if len(parts) == 2 and not parts[1].startswith("_orig_mod."):
            new_k = f"{parts[0]}._orig_mod.{parts[1]}"
        else:
            new_k = k
        new_d[new_k] = v

    save_file(new_d, fixed_path, metadata={"format": "pt"})
    print(f"Saved fixed weights to {fixed_path}")
    return fixed_path

if __name__ == "__main__":
    fix_weights()
