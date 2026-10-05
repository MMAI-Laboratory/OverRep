import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoConfig, GenerationConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM

from overrep.models.registry import register_model
from overrep.models.utils import load_state_dict


def get_layer_number(key):
    m = re.search(r"model\.layers\.(\d+)\.", key)
    if m:
        layer_idx = int(m.group(1))
    else:
        layer_idx = None
    return layer_idx


def reindex_state_dict(state_dict):
    existing_layers = set()

    for key in state_dict.keys():
        idx = get_layer_number(key)
        if idx is not None:
            existing_layers.add(idx)

    sorted_layers = sorted(list(existing_layers))

    indexing_map = {old: new for new, old in enumerate(sorted_layers)}

    new_state_dict = {}
    for old_key, value in state_dict.items():
        if 'model.layers' not in old_key:
            new_state_dict[old_key] = value
            continue

        new_key = re.sub(
            r'^model\.layers\.(\d+)',
            lambda m: f"model.layers.{indexing_map[int(m.group(1))]}",
            old_key
        )
        new_state_dict[new_key] = value

    return new_state_dict, indexing_map


def get_name(config_name):
    if config_name == 'meta-llama/Llama-2-7b-hf':
        name = 'llama2-7B'
    elif config_name == 'meta-llama/Llama-2-13b-hf':
        name = 'llama2-13B'
    elif config_name == 'meta-llama/Llama-3.2-3B':
        name = 'llama3-3B'
    elif config_name == 'meta-llama/Llama-3.1-8B':
        name = 'llama3-8B'
    elif config_name == 'Qwen/Qwen3-4B-Base':
        name = 'qwen3-4B'
    elif config_name == 'Qwen/Qwen3-8B-Base':
        name = 'qwen3-8B'
    elif config_name == 'Qwen/Qwen3-14B-Base':
        name = 'qwen3-14B'
    elif config_name == 'Qwen/Qwen3-30B-A3B-Base':
        name = 'qwen3moe-30B-A3B'
    else:
        raise ValueError
    return name


def get_modules(name):
    if 'llama' in name:
        causal_lm_module = LlamaForCausalLM
        decoder_layer_module = LlamaDecoderLayer
    elif 'qwen3moe' in name:
        causal_lm_module = Qwen3MoeForCausalLM
        decoder_layer_module = Qwen3MoeDecoderLayer
    elif 'qwen3' in name:
        causal_lm_module = Qwen3ForCausalLM
        decoder_layer_module = Qwen3DecoderLayer
    else:
        raise ValueError
    return causal_lm_module, decoder_layer_module


@register_model
def deploy_model(**kwargs):
    config = AutoConfig.from_pretrained(kwargs['config_name'])
    name = get_name(kwargs['config_name'])
    checkpoint = Path(kwargs['checkpoint'])
    causal_lm_module, decoder_layer_module = get_modules(name)

    num_remove_layer = 0
    pruning_configs = list()
    for item in kwargs['include_layer'].split('/'):
        target_layer, interval = item.split(':')
        pruning_configs.append((int(target_layer), int(interval)))
        num_remove_layer = num_remove_layer + int(interval)

    config.num_hidden_layers = config.num_hidden_layers - num_remove_layer
    if 'layer_types' in config:
        n = config.num_hidden_layers
        config.layer_types = ["full_attention"] * n
        config.max_window_layers = n

    state_dict = load_state_dict(name, tie_word_embeddings=config.tie_word_embeddings)

    pop_list = list()
    for key in state_dict.keys():
        layer_idx = get_layer_number(key)
        if layer_idx is None:
            continue


        for target_layer, interval in pruning_configs:
            if layer_idx in range(target_layer, target_layer + interval + 2):
                pop_list.append(key)

    for p in pop_list:
        state_dict.pop(p)

    for target_layer, _ in pruning_configs:
        path = checkpoint / f'layer_{target_layer}.safetensors'
        _state = load_file(path) if path.exists() else torch.load(path.with_suffix('.pth'), map_location='cpu')
        state_dict.update(_state)

    state_dict, indexing_map = reindex_state_dict(state_dict)

    with init_empty_weights():
        model = causal_lm_module(config)
        for target_layer, _ in pruning_configs:
            model.model.layers[indexing_map[target_layer]] = decoder_layer_module(config, indexing_map[target_layer])

    model.load_state_dict(state_dict, assign=True)
    model.tie_weights()

    model.generation_config = GenerationConfig.from_pretrained(kwargs['config_name'])
    return model
