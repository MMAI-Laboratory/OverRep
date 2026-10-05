import json
import os
import re
from glob import glob

from safetensors import safe_open


def shard_sort_key(p: str):
    m = re.search(r"_B(\d+)\.safetensors$", os.path.basename(p))
    return int(m.group(1)) if m else os.path.basename(p)


def build_manifest(index, feature_dir, manifest_dir):
    PATTERN = os.path.join(feature_dir, f"hidden_states_layer{index}_*.safetensors")
    OUT = os.path.join(manifest_dir, f"manifest_layer{index}.json")
    FEATURE_KEY = 'hidden_states'
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)

    files = glob(PATTERN)
    files = sorted(files, key=shard_sort_key)
    if not files:
        return None, None

    manifest = {
        "feature_key": FEATURE_KEY,
        "shards": [],
        "cumulative_sizes": [],
    }

    total = 0
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            x = f.get_tensor(FEATURE_KEY)
            if x.dim() < 1:
                raise ValueError(f"{fp}:{FEATURE_KEY} must have batch dim, got shape {tuple(x.shape)}")
            n = x.shape[0]
        total += n
        manifest["shards"].append({"path": os.path.abspath(fp), "num_batches": n})
        manifest["cumulative_sizes"].append(total)

    manifest["num_batches"] = total

    with open(OUT, "w") as w:
        json.dump(manifest, w, indent=2)

    return len(manifest['shards']), total
