import math
from functools import partial

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    logging,
)

logger = logging.get_logger(__name__)


def _dispatch_model(model, dtype=torch.float32):
    if torch.cuda.device_count() == 2:
        custom_device_map = {
            'rotary_emb': 0,
            'position_ids': 0,
            'decoder_layer': 0,
            'last_decoder_layer': 1,
            'norm': 1,
        }
    elif torch.cuda.device_count() == 4:
        custom_device_map = {
            'rotary_emb': 0,
            'position_ids': 0,
            'decoder_layer.input_layernorm': 0,
            'decoder_layer.self_attn': 0,
            'decoder_layer.post_attention_layernorm': 0,
            'decoder_layer.mlp': 1,
            'last_decoder_layer.input_layernorm': 2,
            'last_decoder_layer.self_attn': 2,
            'last_decoder_layer.post_attention_layernorm': 2,
            'last_decoder_layer.mlp': 3,
            'norm': 3,
        }
    else:
        raise NotImplementedError
    for k, v in model.state_dict().items():
        custom_device_map[k] = 0
    return dispatch_model(model, device_map=custom_device_map)


def auto_dispatch_model(model, dtype):
    no_split_module_classes = ['LlamaDecoderLayer', 'Qwen3DecoderLayer']
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_module_classes,
        dtype=dtype,
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        dtype=dtype,
    )

    return dispatch_model(model, device_map=device_map)


def convert_model_dtype(model, dtype=torch.float32):
    for param in model.parameters():
        param.data = param.data.to(dtype)

    for buffer_name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        num_cycles: float = 0.5,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
        max_learning_rate: float,
        min_learning_rate: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    _lambda = max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return (min_learning_rate + _lambda * (max_learning_rate - min_learning_rate)) / max_learning_rate


def configure_cuda(enable_tf32: bool = True) -> None:
    import torch as _torch
    if hasattr(_torch.backends.cuda.matmul, "fp32_precision"):
        _torch.backends.cuda.matmul.fp32_precision = "tf32" if enable_tf32 else "ieee"
    _torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    _torch.backends.cudnn.allow_tf32 = enable_tf32
    if hasattr(_torch.backends.cudnn, "conv") and hasattr(_torch.backends.cudnn.conv, "fp32_precision"):
        _torch.backends.cudnn.conv.fp32_precision = "tf32" if enable_tf32 else "ieee"
