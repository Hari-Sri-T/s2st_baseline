import os
from safetensors.torch import load_file, save_file

def fix_weights():
    path = "local_indicf5/model.safetensors"
    print("Fixing weights for", path)
    d = load_file(path)
    new_d = {}
    for k, v in d.items():
        if "vocoder." in k:
            continue
        new_k = k.replace("_orig_mod.", "")
        new_d[new_k] = v
    new_path = path.replace(".safetensors", "_fixed.safetensors")
    save_file(new_d, new_path, metadata={"format": "pt"})
    print("Saved fixed weights to", new_path)

if __name__ == "__main__":
    fix_weights()
