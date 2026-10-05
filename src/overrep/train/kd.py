"""OverRep-KD: logit distillation against the teacher streams cached by `overrep-cache-kd`."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, set_seed
from transformers.trainer_pt_utils import get_model_param_count

from overrep.common.sharded_dataset import SafeTensorShards
from overrep.common.utils import convert_model_dtype
from overrep.common.utils import get_cosine_schedule_with_warmup as utils_cosine
from overrep.models import create_model, run_module_function
from overrep.models.vanishing_activation import va_step

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger.setLevel(logging.INFO)


class PairShards(Dataset):
    def __init__(self, prefix_manifest, final_manifest):
        self.a = SafeTensorShards(prefix_manifest)
        self.b = SafeTensorShards(final_manifest)
        assert len(self.a) == len(self.b), (len(self.a), len(self.b))

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return {'h_prefix': self.a[i], 'h_final': self.b[i]}


def chunked_kl_from_hidden(reparam_out, h_final, lm_head, chunk_size, temperature):
    B, S, _ = reparam_out.shape
    total = reparam_out.new_zeros((), dtype=torch.float32)
    n_tok = B * S
    head_dtype = lm_head.weight.dtype
    for s in range(0, S, chunk_size):
        sl = slice(s, min(s + chunk_size, S))
        with torch.no_grad():
            t = lm_head(h_final[:, sl].to(head_dtype)).float() / temperature
            p_t = F.softmax(t, dim=-1)
            log_p_t = F.log_softmax(t, dim=-1)
        log_p_s = F.log_softmax(lm_head(reparam_out[:, sl].to(head_dtype)).float() / temperature, dim=-1)
        total = total + (p_t * (log_p_t - log_p_s)).sum()
    return total / n_tok * (temperature ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--output_dir', default='output')
    p.add_argument('--config_name', required=True)
    p.add_argument('--model_key', required=True)
    p.add_argument('--target_layer', type=int, required=True)
    p.add_argument('--layer_interval', type=int, required=True)
    p.add_argument('--cache_root', required=True, help='Cache root produced by overrep-cache-kd.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--chunk_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--adam8bit', action='store_true')
    args = p.parse_args()

    lr, wd = 1e-4, 1e-2
    bs, accum = args.batch_size, args.gradient_accumulation_steps
    out_dir = Path(args.output_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    transformers.utils.logging.set_verbosity(logging.WARNING)
    set_seed(args.seed)
    dev = torch.device('cuda')
    T, interval = args.target_layer, args.layer_interval

    cache = Path(args.cache_root)
    ds = PairShards(cache / f'prefix_T{T}' / 'manifest.json', cache / 'teacher_final' / 'manifest.json')
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=4)
    logger.info(f"[1/4] Cached dataset {len(ds):,} | bs={bs} accum={accum} (total {bs * accum}) lr={lr} ep={args.epochs}")

    base = AutoModelForCausalLM.from_pretrained(args.config_name, torch_dtype=torch.bfloat16)
    lm_head = base.lm_head.to(dev)
    lm_head.requires_grad_(False)
    del base
    torch.cuda.empty_cache()

    logger.info(f"[2/4] Building reparam: {args.model_key} T={T} I={interval} (va+rep_norm)")
    reparam = create_model(args.model_key, target_layer=T, config_name=args.config_name,
                           layer_interval=interval, va=True, max_length=1024, rep_norm=True)
    convert_model_dtype(reparam)
    reparam = reparam.to(dev)
    train_p = get_model_param_count(reparam, trainable_only=True)
    logger.info(f"  Reparam trainable: {train_p / 1e6:.2f}M")

    no_decay = ["bias", "layernorm"]
    groups = [
        {"params": [pp for n, pp in reparam.named_parameters()
                    if not any(nd in n for nd in no_decay) and pp.requires_grad], "weight_decay": wd},
        {"params": [pp for n, pp in reparam.named_parameters()
                    if any(nd in n for nd in no_decay) and pp.requires_grad], "weight_decay": 0.0},
    ]
    total_steps = (len(dl) // accum) * args.epochs
    if args.adam8bit:
        optimizer = bnb.optim.AdamW8bit(groups, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, fused=True)
    scheduler = utils_cosine(optimizer, int(total_steps * 0.01), total_steps, lr, 1e-5)
    run_module_function(reparam, total_steps=total_steps, patient_steps_ratio=0.2)

    logger.info(f"[3/4] Cached-KD training: {total_steps} opt steps")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    global_step, nan_count = 0, 0
    logging_step = max(1, (len(dl) // accum) // 16)

    for epoch in range(args.epochs):
        reparam.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl):
            h_pref = batch['h_prefix'].to(dev, non_blocking=True)
            h_fin = batch['h_final'].to(dev, non_blocking=True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = reparam(h_pref)
            loss = chunked_kl_from_hidden(out, h_fin, lm_head, args.chunk_size, args.temperature)

            if not torch.isfinite(loss):
                nan_count += 1
                logger.warning(f"NaN/Inf ep{epoch + 1} step {step} (count={nan_count}), skip.")
                del out
                continue

            (loss / accum).backward()
            running += loss.item()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_([pp for pp in reparam.parameters() if pp.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                va_step(reparam)
                global_step += 1
                if global_step % logging_step == 0 or global_step <= 3:
                    logger.info(f"[Epoch {epoch + 1}] step {global_step} | KD loss: {running / (step + 1):.4f} "
                                f"| lr: {scheduler.get_last_lr()[0]:.2e} "
                                f"| peak: {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
        logger.info(f"[Epoch {epoch + 1}] finished | avg KD loss: {running / len(dl):.4f} | nan: {nan_count}")

    wall = time.time() - t0
    logger.info("[4/4] Saving merged reparam state")
    torch.save({k: v.cpu() for k, v in reparam.state_dict().items()}, out_dir / f'layer_{T}.pth')

    summary = dict(
        name=args.name, model_key=args.model_key, target_layer=T, layer_interval=interval,
        trainable_params_M=round(train_p / 1e6, 2), lr=lr, weight_decay=wd,
        batch_size=bs, grad_accum=accum, total_batch=bs * accum, epochs=args.epochs,
        temperature=args.temperature, va=True, rep_norm=True, seed=args.seed,
        optimizer='AdamW8bit' if args.adam8bit else 'AdamW(fused)',
        cache_root=str(cache),
        final_epoch_avg_kd_loss=running / len(dl), nan_microbatches=nan_count,
        wall_clock_sec=round(wall, 1), n_gpus=1,
        peak_mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved -> {out_dir} | {json.dumps(summary)}")


if __name__ == '__main__':
    main()
