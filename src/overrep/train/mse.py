"""OverRep-MSE trainer: layer-wise feature distillation on cached hidden states."""
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from transformers.trainer_pt_utils import get_model_param_count

from overrep.common.constants import ENTITY
from overrep.common.sharded_dataset import SafeTensorShards
from overrep.common.utils import _dispatch_model, convert_model_dtype, get_cosine_schedule_with_warmup
from overrep.models import create_model, run_module_function
from overrep.models.vanishing_activation import VanishingActivation, va_step

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def configure_cuda(enable_tf32: bool = True) -> None:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable_tf32 else "ieee"
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable_tf32 else "ieee"
    torch.backends.cudnn.allow_tf32 = enable_tf32

    torch.backends.cudnn.benchmark = True


configure_cuda()


@dataclass
class InfoArguments:
    name: str = field(default=None, metadata={"help": "Experiment Name"})
    wandb: bool = field(default=False, metadata={"help": "Turn on wandb."})
    project: str = field(default='OverRep', metadata={"help": "The project name of wandb"})
    output_dir: str = field(default='output', metadata={"help": "Where to save output"})
    should_log: bool = field(default=False, metadata={"help": "Whether to log"})


@dataclass
class ModelArguments:
    custom: bool = field(default=False, metadata={"help": "Is use custom models"})
    model_name_or_path: str = field(default=None, metadata={"help": "Path to checkpoint for model"})
    config_name: str = field(default=None, metadata={"help": "Which config to use"})
    va: bool = field(default=False, metadata={"help": "Vanishing Activation"})
    adam8bit: bool = field(default=False, metadata={"help": "Paged 8-bit AdamW (large reparam on a single GPU)"})
    rep_norm: bool = field(default=False, metadata={"help": "Apply Rep Normalization"})


@dataclass
class TrainingArguments:
    target_layer: int = field(default=None, metadata={"help": "Which layer of model to KD"})
    layer_interval: int = field(default=None, metadata={"help": "Interval of layer"})
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    patient_steps_ratio: float = field(default=0.2, metadata={"help": "Patient steps of vanishing activation"})
    mixed_precision: str = field(default="no", metadata={"help": "Which dtype to use"})
    max_length: int = field(default=1024, metadata={"help": "Max length"})

    dataset: str = field(default=None, metadata={"help": "Which dataset to use"})
    batch_size: int = field(default=8, metadata={"help": "Batch size"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of accumulation steps"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers"})

    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    min_learning_rate: float = field(default=1e-5, metadata={"help": "Min learning rate"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay"})
    seed: int = field(default=42, metadata={"help": "Random seed"})


class DictDataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
        self.keys = list(dataset_dict.keys())

    def __len__(self):
        return len(self.dataset_dict[self.keys[0]])

    def __getitem__(self, idx):
        return {key: self.dataset_dict[key][idx] for key in self.keys}


class _RunningMean:
    __slots__ = ("_sum", "_count")

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        self._sum += float(value)
        self._count += 1

    def compute(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0


def layerwise_dataloader(dataset_root, target_layer, layer_interval, batch_size, num_workers, shuffle) \
        -> tuple[DataLoader[Any], DictDataset]:
    ids_path = os.path.join(dataset_root, f'manifest_layer{target_layer - 1}.json')
    ods_path = os.path.join(dataset_root, f'manifest_layer{target_layer + layer_interval + 1}.json')

    ds = DictDataset({
        'input_features': SafeTensorShards(ids_path),
        'output_features': SafeTensorShards(ods_path),
    })
    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                     pin_memory=True, persistent_workers=num_workers > 0)
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **dl_kwargs), ds


def _build_model(model_args: ModelArguments, training_args: TrainingArguments):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()

    if not model_args.model_name_or_path:
        return AutoModelForCausalLM.from_config(config)

    if not model_args.custom:
        return AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

    model = create_model(
        model_args.model_name_or_path,
        target_layer=training_args.target_layer,
        config_name=model_args.config_name,
        layer_interval=training_args.layer_interval,
        va=model_args.va,
        max_length=training_args.max_length,
        rep_norm=model_args.rep_norm,
    )
    convert_model_dtype(model)
    if torch.cuda.device_count() > 1:
        _dispatch_model(model)
    return model


def _build_optimizer(model, training_args: TrainingArguments):
    no_decay = ["bias", "layernorm"]
    parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        parameters, lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay, fused=True,
    )


def _maybe_init_wandb(info_args, model_args, training_args, dataset_size: int) -> bool:
    if not info_args.wandb:
        return False
    wandb.init(
        project=info_args.project,
        entity=ENTITY,
        config={
            'info_args': asdict(info_args),
            'model_args': asdict(model_args),
            'training_args': asdict(training_args),
            'dataset_size': dataset_size,
        },
        name=info_args.name,
        settings=wandb.Settings(_disable_stats=True),
    )
    return True


def _train_one_epoch(
        *, model, dl, optimizer, scheduler,
        accelerator, model_args, epoch, total_epochs, batch_size, use_wandb,
):
    model.train()
    running = _RunningMean()
    logging_iter = int(os.environ.get('LOG_EVERY', max(1, len(dl) // 16)))
    s = perf_counter()

    for i, item in enumerate(dl):
        x = item['input_features']
        y = item['output_features']
        with accelerator.accumulate(model):
            out_features = model(x)
            loss = F.mse_loss(out_features.to(torch.float32), y.to(torch.float32))
            accelerator.backward(loss)
            del out_features

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if model_args.va:
                    va_step(model)

        running.update(loss.detach().float().item())

        duration = perf_counter() - s
        s = perf_counter()
        if i % logging_iter == 0:
            mean_loss = running.compute()
            running.reset()

            lrl = [pg['lr'] for pg in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            alpha = 1.0
            if model_args.va:
                for m in model.modules():
                    if isinstance(m, VanishingActivation):
                        alpha = m.alpha
                        break

            nb_remain = len(dl) - i - 1 + (total_epochs - epoch - 1) * len(dl)
            eta_seconds = duration * nb_remain
            logger.debug(
                f'{"Train":>5}: {epoch:>3} [{i:>4d}/{len(dl) - 1}] '
                f'({100. * i / (len(dl) - 1):>3.0f}%)]  '
                f'Loss: {mean_loss:#.3g}  '
                f'LR: {lr:.3e}  '
                f'TP: {batch_size / duration:>7.2f}/s  '
                f'ETA: {timedelta(seconds=int(eta_seconds))}  '
            )
            if use_wandb:
                wandb.log({'loss': mean_loss, 'lr': lr, 'alpha': alpha})


def _validate(model, val_dl) -> float:
    model.eval()
    running = _RunningMean()
    with torch.inference_mode():
        for item in val_dl:
            out_features = model(item['input_features'])
            loss = F.mse_loss(out_features, item['output_features'].to(torch.float32))
            running.update(loss.detach().float().item())
    return running.compute()


def _save_state_dict(model, accelerator, output_path: Path) -> None:
    state_dict = accelerator.unwrap_model(model).state_dict()
    torch.save({k: v.cpu() for k, v in state_dict.items()}, output_path)


def main():
    parser = HfArgumentParser((InfoArguments, ModelArguments, TrainingArguments))
    info_args, model_args, training_args = parser.parse_args_into_dataclasses()
    assert model_args.custom is not None and model_args.config_name is not None

    output_dir = Path(info_args.output_dir) / info_args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if info_args.should_log else logging.INFO
    if info_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(logging.WARNING)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    model = _build_model(model_args, training_args)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.debug(f"Total Model Size: {n_params / 1e6:.2f}M")
    logger.debug(f"Number of trainable parameters = {get_model_param_count(model, trainable_only=True) / 1e6:.2f}M")

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
    )

    dl, ds = layerwise_dataloader(
        training_args.dataset + '_train', training_args.target_layer, training_args.layer_interval,
        training_args.batch_size, training_args.num_workers, True)
    val_dl, _ = layerwise_dataloader(
        training_args.dataset + '_test', training_args.target_layer, training_args.layer_interval,
        training_args.batch_size, training_args.num_workers, False)

    total_steps = len(dl) * training_args.epochs // training_args.gradient_accumulation_steps

    optimizer = _build_optimizer(model, training_args)
    if model_args.adam8bit:
        import bitsandbytes as bnb
        groups = optimizer.param_groups
        optimizer = bnb.optim.PagedAdamW8bit(
            [{'params': g['params'], 'weight_decay': g['weight_decay']} for g in groups],
            lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    warmup_steps = int(total_steps * 0.01)
    if training_args.min_learning_rate > training_args.learning_rate:
        training_args.min_learning_rate = training_args.learning_rate * 0.1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
        training_args.learning_rate, training_args.min_learning_rate,
    )

    use_wandb = _maybe_init_wandb(info_args, model_args, training_args, len(ds))

    model, optimizer, dl, val_dl, scheduler = accelerator.prepare(model, optimizer, dl, val_dl, scheduler)

    if model_args.va:
        run_module_function(
            model, total_steps=total_steps, patient_steps_ratio=training_args.patient_steps_ratio,
        )

    for epoch in range(training_args.epochs):
        _train_one_epoch(
            model=model, dl=dl, optimizer=optimizer, scheduler=scheduler,
            accelerator=accelerator,
            model_args=model_args, epoch=epoch, total_epochs=training_args.epochs,
            batch_size=training_args.batch_size, use_wandb=use_wandb,
        )
        val_loss = _validate(model, val_dl)
        logger.debug(f"[epoch {epoch}] val_loss {val_loss:#.4g}")
        if use_wandb:
            wandb.log({'val_loss': val_loss})
        if os.environ.get('SAVE_EVERY_EPOCH') == '1':
            _save_state_dict(model, accelerator, output_dir / f'layer_{training_args.target_layer}.pth')
            logger.debug(f"[epoch {epoch}] checkpoint saved -> layer_{training_args.target_layer}.pth")

    _save_state_dict(model, accelerator, output_dir / f'layer_{training_args.target_layer}.pth')

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
