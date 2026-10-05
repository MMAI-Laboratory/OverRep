"""Cache per-layer hidden states for OverRep-MSE training.

    overrep-cache --model Qwen/Qwen3-14B-Base \
        --dataset data/tokenized/finewebedu_qwen3_train --layers 17 29 \
        --out data/finewebedu_qwen3-14b_manifest_train --batch_size 2

Writes `{out}/manifest_layer{L}.json` and `{out}/shards_layer{L}/shard_*.safetensors`, where `L` is the
layer whose output is stored; the last layer is stored after the final norm, as in `output_hidden_states`.
For mask `T:I`, training reads layers `T-1` (input) and `T+I+1` (target).
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers.data.data_collator import torch_default_data_collator


class ShardWriter:
    def __init__(self, root: Path, layer: int, feature_key='hidden', shard_batches=32):
        self.dir = root / f'shards_layer{layer}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / f'manifest_layer{layer}.json'
        self.fkey = feature_key
        self.shard_batches = shard_batches
        self.buf, self.shards, self.cums, self.total = [], [], [], 0

    def add(self, t: torch.Tensor):
        self.buf.append(t)
        if len(self.buf) >= self.shard_batches:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        x = torch.cat(self.buf, dim=0)
        p = self.dir / f'shard_{len(self.shards):05d}.safetensors'
        save_file({self.fkey: x}, str(p))
        self.total += x.shape[0]
        self.shards.append({'path': str(p), 'num': x.shape[0]})
        self.cums.append(self.total)
        self.buf = []

    def finalize(self):
        self.flush()
        with open(self.manifest_path, 'w') as f:
            json.dump({'shards': self.shards, 'cumulative_sizes': self.cums,
                       'num_batches': self.total, 'feature_key': self.fkey}, f)
        print(f'  {self.manifest_path}: {self.total} samples, {len(self.shards)} shards', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--layers', type=int, nargs='+', required=True, help='Layer indices whose output is dumped.')
    p.add_argument('--out', required=True, help='Manifest root, including the split suffix.')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--limit', type=int, default=None, help='Use only the first N samples of the dataset.')
    args = p.parse_args()

    out = Path(args.out)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()
    model.requires_grad_(False)

    writers = {ly: ShardWriter(out, ly) for ly in args.layers}
    captured = {}

    def grab(ly):
        def fn(m, i, o):
            captured[ly] = o[0] if isinstance(o, tuple) else o
        return fn

    last = model.config.num_hidden_layers - 1
    hooks = [(model.model.norm if ly == last else model.model.layers[ly]).register_forward_hook(grab(ly))
             for ly in args.layers]

    ds = load_from_disk(args.dataset)
    if args.limit:
        ds = ds.select(range(args.limit))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=torch_default_data_collator, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            model(input_ids=batch['input_ids'].to(model.device),
                  attention_mask=batch['attention_mask'].to(model.device) if 'attention_mask' in batch else None)
            for ly, w in writers.items():
                w.add(captured[ly].to('cpu', torch.bfloat16))
            if i % 200 == 0:
                print(f'{i}/{len(dl)}', flush=True)

    for h in hooks:
        h.remove()
    for w in writers.values():
        w.finalize()
    print('DUMP DONE', flush=True)


if __name__ == '__main__':
    main()
