"""Cache the teacher streams for OverRep-KD.

    overrep-cache-kd --model meta-llama/Llama-3.2-3B \
        --dataset data/tokenized/finewebedu_llama3_train \
        --targets 17 10 --out data/kdcache_llama3-3b --batch_size 8

Writes `{out}/{stream}/manifest.json` for each stream:
  - `prefix_T{T}`   : output of layer `T-1` (input of the recovery blocks)
  - `teacher_final` : output of the final norm (input of `lm_head`)
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
    def __init__(self, root: Path, feature_key='hidden', shard_batches=16):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
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
        p = self.root / f'shard_{len(self.shards):05d}.safetensors'
        save_file({self.fkey: x}, str(p))
        self.total += x.shape[0]
        self.shards.append({'path': str(p), 'num': x.shape[0]})
        self.cums.append(self.total)
        self.buf = []

    def finalize(self):
        self.flush()
        with open(self.root / 'manifest.json', 'w') as f:
            json.dump({'shards': self.shards, 'cumulative_sizes': self.cums,
                       'num_batches': self.total, 'feature_key': self.fkey}, f)
        print(f'  {self.root}: {self.total} samples, {len(self.shards)} shards')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--targets', type=int, nargs='+', required=True, help='Reparam target layers T; the output of layer T-1 is dumped for each.')
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    out = Path(args.out)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)

    writers = {f'prefix_T{t}': ShardWriter(out / f'prefix_T{t}') for t in args.targets}
    writers['teacher_final'] = ShardWriter(out / 'teacher_final')

    captured = {}
    hooks = []

    def _grab(key):
        def fn(m, i, o):
            captured[key] = o[0] if isinstance(o, tuple) else o
        return fn

    for t in args.targets:
        hooks.append(model.model.layers[t - 1].register_forward_hook(_grab(f'prefix_T{t}')))
    hooks.append(model.model.norm.register_forward_hook(_grab('teacher_final')))

    ds = load_from_disk(args.dataset)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=torch_default_data_collator, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            model(input_ids=batch['input_ids'].cuda(),
                  attention_mask=batch['attention_mask'].cuda() if 'attention_mask' in batch else None)
            for k, w in writers.items():
                w.add(captured[k].to('cpu', torch.bfloat16))
            if i % 100 == 0:
                print(f'{i}/{len(dl)}', flush=True)

    for h in hooks:
        h.remove()
    for w in writers.values():
        w.finalize()
    print('DUMP DONE')


if __name__ == '__main__':
    main()
