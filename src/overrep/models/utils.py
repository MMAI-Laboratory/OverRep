import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

MODEL_REPOS = {
    'llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'llama2-13B': 'meta-llama/Llama-2-13b-hf',
    'llama3-3B': 'meta-llama/Llama-3.2-3B',
    'llama3-8B': 'meta-llama/Llama-3.1-8B',
    'qwen3-4B': 'Qwen/Qwen3-4B-Base',
    'qwen3-8B': 'Qwen/Qwen3-8B-Base',
    'qwen3-14B': 'Qwen/Qwen3-14B-Base',
    'qwen3moe-30B-A3B': 'Qwen/Qwen3-30B-A3B-Base',
}


class LazyStateDict:
    """Loads tensors from safetensors shards on demand. Enabled with `OVERREP_LAZY_SD=1`."""

    def __init__(self, shard_paths):
        self._key_to_path = {}
        for p in shard_paths:
            with safe_open(str(p), framework="pt", device="cpu") as f:
                for k in f.keys():
                    self._key_to_path[k] = str(p)
        self._overrides = {}

    def __getitem__(self, key):
        if key in self._overrides:
            return self._overrides[key]
        path = self._key_to_path[key]
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def __contains__(self, key):
        return key in self._overrides or key in self._key_to_path

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return list(dict.fromkeys(list(self._key_to_path) + list(self._overrides)))

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, *default):
        try:
            value = self[key]
        except KeyError:
            if default:
                return default[0]
            raise
        self._overrides.pop(key, None)
        self._key_to_path.pop(key, None)
        return value

    def update(self, other):
        self._overrides.update(other)


def resolve_snapshot_dir(name):
    from huggingface_hub import snapshot_download

    repo_id = MODEL_REPOS.get(name, name)
    patterns = ["*.safetensors", "*.safetensors.index.json", "config.json"]
    try:
        snapshot_dir = snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True)
    except Exception:
        snapshot_dir = snapshot_download(repo_id, allow_patterns=patterns)

    snapshot_dir = Path(snapshot_dir)
    if not list(snapshot_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"No safetensors shards for {repo_id} under {snapshot_dir}")
    return snapshot_dir


def load_state_dict(name, tie_word_embeddings=False):
    shard_paths = sorted(resolve_snapshot_dir(name).glob("model*.safetensors"))
    if os.getenv("OVERREP_LAZY_SD") == "1":
        state_dict = LazyStateDict(shard_paths)
    else:
        state_dict = {}
        for file_path in shard_paths:
            state_dict.update(load_file(file_path))

    pop_list = [k for k in state_dict.keys() if 'model.layers' in k and 'rotary_emb' in k]
    for p in pop_list:
        state_dict.pop(p)

    if tie_word_embeddings:
        state_dict.update({'lm_head.weight': state_dict['model.embed_tokens.weight']})
    return state_dict
