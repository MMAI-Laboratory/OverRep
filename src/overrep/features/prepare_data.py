"""Tokenize FineWeb-Edu into fixed-length train/test blocks.

    overrep-prepare --model meta-llama/Llama-3.2-3B --out data/tokenized/finewebedu_llama3

Writes `{out}_train` and `{out}_test` with `datasets.save_to_disk`. Run once per backbone tokenizer.
"""
import argparse
from itertools import chain
from pathlib import Path

import datasets
from transformers import AutoTokenizer

RAW_COLUMNS_TO_DROP = ['id', 'dump', 'url', 'file_path', 'language',
                       'language_score', 'token_count', 'score', 'int_score']


def group_texts(examples, block_size):
    concatenated = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    return {k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()}


def tokenize_and_group(raw_dataset, tokenizer, block_size, num_proc):
    tokenized = raw_dataset.map(
        lambda examples: tokenizer(examples['text']),
        batched=True,
        remove_columns=['text'],
        num_proc=num_proc,
    )
    return tokenized.map(lambda examples: group_texts(examples, block_size),
                         batched=True, num_proc=num_proc)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, help='Tokenizer to use (Hugging Face repo id).')
    p.add_argument('--out', required=True, help='Output prefix; `_train` / `_test` are appended.')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--train-samples', type=int, default=12000)
    p.add_argument('--test-samples', type=int, default=4000)
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--num-proc', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    if Path(f'{out}_train').exists():
        raise SystemExit(f'{out}_train already exists — remove it or pick another --out.')

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, max_length=args.block_size)

    raw = datasets.load_dataset(args.dataset, name=args.dataset_config, split='train')
    raw = raw.shuffle(seed=args.seed).select(range(args.train_samples + args.test_samples))
    raw = raw.remove_columns([c for c in RAW_COLUMNS_TO_DROP if c in raw.column_names])
    raw = raw.train_test_split(train_size=args.train_samples, test_size=args.test_samples, seed=args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        ds = tokenize_and_group(raw[split], tokenizer, args.block_size, args.num_proc)
        ds.save_to_disk(f'{out}_{split}')
        print(f'{out}_{split}: {len(ds)} blocks x {args.block_size} tokens')


if __name__ == '__main__':
    main()
