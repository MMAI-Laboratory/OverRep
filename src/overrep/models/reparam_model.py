from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)
from transformers.processing_utils import Unpack

from overrep.models.registry import register_model
from overrep.models.reparam_module_llama import LlamaRepAttention, LlamaRepMLP, LlamaRepRMSNorm
from overrep.models.reparam_module_qwen import Qwen3RepAttention, Qwen3RepMLP, Qwen3RepRMSNorm
from overrep.models.reparam_module_qwen3_moe import (
    Qwen3MoeRepAttention,
    Qwen3MoeRepRMSNorm,
    Qwen3MoeRepSparseMoeBlock,
)
from overrep.models.utils import load_state_dict


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        return hidden_states


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        return hidden_states


class CustomQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        return hidden_states


def get_specific_module(model_name: str) -> dict:
    MODULES = {
        'llama': (CustomLlamaDecoderLayer, LlamaRotaryEmbedding, LlamaRMSNorm, LlamaRepAttention, LlamaRepMLP,
                  LlamaRepRMSNorm),
        'qwen3': (CustomQwen3DecoderLayer, Qwen3RotaryEmbedding, Qwen3RMSNorm, Qwen3RepAttention, Qwen3RepMLP,
                  Qwen3RepRMSNorm),
        'qwen3moe': (CustomQwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding, Qwen3MoeRMSNorm, Qwen3MoeRepAttention,
                     Qwen3MoeRepSparseMoeBlock, Qwen3MoeRepRMSNorm),
    }
    KEYS = ('decoder', 'rotary_embedding', 'norm', 'rep_attn', 'rep_mlp', 'rep_norm')

    if model_name.startswith('llama'):
        return dict(zip(KEYS, MODULES['llama']))
    elif model_name.startswith('qwen3moe'):
        return dict(zip(KEYS, MODULES['qwen3moe']))
    elif model_name.startswith('qwen3'):
        return dict(zip(KEYS, MODULES['qwen3']))
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")


def _input_layernorm_key(i):
    return f'model.layers.{i}.input_layernorm.weight'


def _post_attention_layernorm_key(i):
    return f'model.layers.{i}.post_attention_layernorm.weight'


def _mlp_key(i, name):
    return f'model.layers.{i}.mlp.{name}.weight'


def _attn_key(i, name):
    return f'model.layers.{i}.self_attn.{name}.weight'


def _is_moe_state_dict(state_dict):
    return 'model.layers.0.mlp.experts.0.gate_proj.weight' in state_dict


def _moe_attn_weights(state_dict, idx):
    def attn(name):
        return state_dict[f'model.layers.{idx}.self_attn.{name}']

    return [attn('q_proj.weight'), attn('k_proj.weight'), attn('v_proj.weight'), attn('o_proj.weight'),
            attn('q_norm.weight'), attn('k_norm.weight')]


def _moe_mlp_weights(state_dict, idx):
    prefix = f'model.layers.{idx}.mlp.'
    gate_weight = state_dict[prefix + 'gate.weight']

    expert_weights = []
    j = 0
    while prefix + f'experts.{j}.gate_proj.weight' in state_dict:
        expert_weights.append((state_dict[prefix + f'experts.{j}.gate_proj.weight'],
                               state_dict[prefix + f'experts.{j}.up_proj.weight'],
                               state_dict[prefix + f'experts.{j}.down_proj.weight']))
        j += 1

    return [gate_weight, expert_weights]


def pretrained_weight_by_adaptive(state_dict, start_idx, end_idx):
    input_layernorm_weights = state_dict[_input_layernorm_key(end_idx)]
    post_attention_layernorm_weights = state_dict[_post_attention_layernorm_key(end_idx)]

    if _is_moe_state_dict(state_dict):
        return (input_layernorm_weights, post_attention_layernorm_weights,
                _moe_attn_weights(state_dict, start_idx), _moe_mlp_weights(state_dict, end_idx))

    use_qn = False
    for k, _ in state_dict.items():
        if 'q_norm.weight' in k:
            use_qn = True
            break

    if use_qn:
        attn_weights = [state_dict[_attn_key(start_idx, name)] for name in
                        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm']]
    else:
        attn_weights = [state_dict[_attn_key(start_idx, name)] for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']]
    mlp_weights = [state_dict[_mlp_key(end_idx, name)] for name in ['gate_proj', 'up_proj', 'down_proj']]

    return input_layernorm_weights, post_attention_layernorm_weights, attn_weights, mlp_weights


def pretrained_weight_by_idx(state_dict, idx):
    input_layernorm_weights = state_dict[_input_layernorm_key(idx)]
    post_attention_layernorm_weights = state_dict[_post_attention_layernorm_key(idx)]

    if _is_moe_state_dict(state_dict):
        return (input_layernorm_weights, post_attention_layernorm_weights,
                _moe_attn_weights(state_dict, idx), _moe_mlp_weights(state_dict, idx))

    use_qn = False
    for k, _ in state_dict.items():
        if 'q_norm.weight' in k:
            use_qn = True
            break

    if use_qn:
        attn_weights = [state_dict[_attn_key(idx, name)] for name in
                        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm']]
    else:
        attn_weights = [state_dict[_attn_key(idx, name)] for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']]
    mlp_weights = [state_dict[_mlp_key(idx, name)] for name in ['gate_proj', 'up_proj', 'down_proj']]

    return input_layernorm_weights, post_attention_layernorm_weights, attn_weights, mlp_weights


class MultiDepthReparamLM(nn.Module):
    def __init__(self, config, name, decoder_layer_module, embed_module, norm_module, mlp_module, attn_module,
                 rep_norm_module, target_layer, layer_interval, width_num_module, depth_num_module, va=False,
                 rep_norm=False, max_length=1024):
        super(MultiDepthReparamLM, self).__init__()
        self.config = config
        self.target_layer = target_layer
        self.rep_norm = rep_norm

        self.is_last_layer = self._validate_target_range(config, target_layer, layer_interval)
        config._attn_implementation_internal = "sdpa"

        self.rotary_emb = embed_module(config)
        self.register_buffer("position_ids", torch.arange(max_length).unsqueeze(0))

        state_dict = load_state_dict(name, config.tie_word_embeddings)

        if isinstance(width_num_module, int):
            width_num_module = (width_num_module,) * 4
        if isinstance(depth_num_module, int):
            depth_num_module = (depth_num_module,) * 4

        weights = pretrained_weight_by_adaptive(state_dict, target_layer, target_layer + layer_interval)
        self.decoder_layer = self._build_decoder_layer(
            decoder_layer_module=decoder_layer_module,
            attn_module=attn_module,
            mlp_module=mlp_module,
            rep_norm_module=rep_norm_module,
            input_layernorm_weights=weights[0],
            post_attention_layernorm_weights=weights[1],
            attn_weights=weights[2],
            mlp_weights=weights[3],
            config=config,
            target_layer=target_layer,
            width_num_module=width_num_module[:2],
            depth_num_module=depth_num_module[:2],
            rep_norm=rep_norm,
            va=va,
            freeze_pretrained=False,
        )

        weights = pretrained_weight_by_idx(state_dict, target_layer + layer_interval + 1)
        self.last_decoder_layer = self._build_decoder_layer(
            decoder_layer_module=decoder_layer_module,
            attn_module=attn_module,
            mlp_module=mlp_module,
            rep_norm_module=rep_norm_module,
            input_layernorm_weights=weights[0],
            post_attention_layernorm_weights=weights[1],
            attn_weights=weights[2],
            mlp_weights=weights[3],
            config=config,
            target_layer=target_layer + 1,
            width_num_module=width_num_module[2:],
            depth_num_module=depth_num_module[2:],
            rep_norm=rep_norm,
            va=va,
            freeze_pretrained=True,
        )
        self.norm = None
        if self.is_last_layer:
            if rep_norm:
                self.norm = self._build_repnorm(rep_norm_module, state_dict['model.norm.weight'], True)
            else:
                self.norm = norm_module(config.hidden_size)
                self.norm.load_state_dict({"weight": state_dict["model.norm.weight"]})
                self.norm.requires_grad_(False)

    @staticmethod
    def _validate_target_range(config, target_layer, layer_interval):
        expected_last_idx = config.num_hidden_layers - 1
        actual_last_idx = target_layer + layer_interval + 1
        if actual_last_idx == expected_last_idx:
            return True
        else:
            return False

    def _build_decoder_layer(
            self,
            decoder_layer_module,
            attn_module,
            mlp_module,
            rep_norm_module,
            input_layernorm_weights,
            post_attention_layernorm_weights,
            attn_weights,
            mlp_weights,
            config,
            target_layer,
            width_num_module,
            depth_num_module,
            rep_norm,
            va,
            freeze_pretrained,
    ):
        layer = decoder_layer_module(config, target_layer)

        layer.self_attn = self._build_attention_module(
            attn_module=attn_module,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            pretrained_weights=attn_weights,
            width_num_module=width_num_module[0],
            depth_num_module=depth_num_module[0],
            va=va,
            freeze_pretrained=freeze_pretrained,
        )

        layer.mlp = self._build_mlp_module(
            mlp_module=mlp_module,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            pretrained_weights=mlp_weights,
            width_num_module=width_num_module[1],
            depth_num_module=depth_num_module[1],
            va=va,
            freeze_pretrained=freeze_pretrained,
        )

        if rep_norm:
            layer.input_layernorm = self._build_repnorm(rep_norm_module, input_layernorm_weights, freeze_pretrained)
            layer.post_attention_layernorm = self._build_repnorm(rep_norm_module, post_attention_layernorm_weights,
                                                                 freeze_pretrained)
        else:
            layer.input_layernorm.load_state_dict({'weight': input_layernorm_weights})
            layer.post_attention_layernorm.load_state_dict({'weight': post_attention_layernorm_weights})

            if freeze_pretrained:
                layer.input_layernorm.requires_grad_(False)
                layer.post_attention_layernorm.requires_grad_(False)

        return layer

    @staticmethod
    def _build_attention_module(
            attn_module,
            hidden_size,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            pretrained_weights,
            width_num_module,
            depth_num_module,
            va,
            freeze_pretrained,
    ):
        attn = attn_module(
            hidden_size,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            width_num_module,
            depth_num_module,
            va,
        )
        attn.init_from_pretrained(*pretrained_weights)
        if freeze_pretrained:
            attn.freeze_pretrained()
        return attn

    @staticmethod
    def _build_mlp_module(
            mlp_module,
            hidden_size,
            intermediate_size,
            hidden_act,
            pretrained_weights,
            width_num_module,
            depth_num_module,
            va,
            freeze_pretrained,
    ):
        mlp = mlp_module(
            hidden_size,
            intermediate_size,
            hidden_act,
            width_num_module,
            depth_num_module,
            va,
        )
        mlp.init_from_pretrained(*pretrained_weights)
        if freeze_pretrained:
            mlp.freeze_pretrained()
        return mlp

    def _configure_layer_norm_trainability(self, layer, freeze_pretrained):
        if self.rep_norm:
            layer.input_layernorm = self._build_rep_rmsnorm(
                layer.input_layernorm.weight,
                freeze_pretrained=freeze_pretrained,
            )
            layer.post_attention_layernorm = self._build_rep_rmsnorm(
                layer.post_attention_layernorm.weight,
                freeze_pretrained=freeze_pretrained,
            )
        elif freeze_pretrained:
            layer.input_layernorm.requires_grad_(False)
            layer.post_attention_layernorm.requires_grad_(False)

    @staticmethod
    def _build_repnorm(rep_norm_module, weight, freeze_pretrained):
        norm = rep_norm_module(weight.shape[0])
        norm.init_from_pretrained(weight)
        if freeze_pretrained:
            norm.freeze_pretrained()
        return norm

    def state_dict(self):
        prefix = f'model.layers.{self.target_layer}.'
        last_prefix = f'model.layers.{self.target_layer + 1}.'

        rename_state = dict()

        for k, v in self.decoder_layer.input_layernorm.state_dict().items():
            rename_state[prefix + 'input_layernorm.' + k] = v
        for k, v in self.decoder_layer.post_attention_layernorm.state_dict().items():
            rename_state[prefix + 'post_attention_layernorm.' + k] = v
        for k, v in self.decoder_layer.mlp.state_dict().items():
            rename_state[prefix + 'mlp.' + k] = v
        for k, v in self.decoder_layer.self_attn.state_dict().items():
            rename_state[prefix + 'self_attn.' + k] = v

        for k, v in self.last_decoder_layer.input_layernorm.state_dict().items():
            rename_state[last_prefix + 'input_layernorm.' + k] = v
        for k, v in self.last_decoder_layer.post_attention_layernorm.state_dict().items():
            rename_state[last_prefix + 'post_attention_layernorm.' + k] = v
        for k, v in self.last_decoder_layer.mlp.state_dict().items():
            rename_state[last_prefix + 'mlp.' + k] = v
        for k, v in self.last_decoder_layer.self_attn.state_dict().items():
            rename_state[last_prefix + 'self_attn.' + k] = v

        if self.norm:
            rename_state['model.norm.weight'] = self.norm.state_dict()['weight']

        return rename_state

    def forward(self, hidden_states):
        position_embeddings = self.rotary_emb(hidden_states, self.position_ids)

        hidden_states = self.decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=self.position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.last_decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=self.position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )
        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class MoeMultiDepthReparamLM(MultiDepthReparamLM):
    def _build_decoder_layer(
            self,
            decoder_layer_module,
            attn_module,
            mlp_module,
            rep_norm_module,
            input_layernorm_weights,
            post_attention_layernorm_weights,
            attn_weights,
            mlp_weights,
            config,
            target_layer,
            width_num_module,
            depth_num_module,
            rep_norm,
            va,
            freeze_pretrained,
    ):
        layer = decoder_layer_module(config, target_layer)

        layer.self_attn = self._build_attention_module(
            attn_module=attn_module,
            hidden_size=config.hidden_size,
            head_dim=getattr(config, 'head_dim', None),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            pretrained_weights=attn_weights,
            width_num_module=width_num_module[0],
            depth_num_module=depth_num_module[0],
            va=va,
            freeze_pretrained=freeze_pretrained,
        )

        layer.mlp = mlp_module(config, width_num_module[1], depth_num_module[1], va)
        layer.mlp.init_from_pretrained(*mlp_weights)
        if freeze_pretrained:
            layer.mlp.freeze_pretrained()

        if rep_norm:
            layer.input_layernorm = self._build_repnorm(rep_norm_module, input_layernorm_weights, freeze_pretrained)
            layer.post_attention_layernorm = self._build_repnorm(rep_norm_module, post_attention_layernorm_weights,
                                                                 freeze_pretrained)
        else:
            layer.input_layernorm.load_state_dict({'weight': input_layernorm_weights})
            layer.post_attention_layernorm.load_state_dict({'weight': post_attention_layernorm_weights})

            if freeze_pretrained:
                layer.input_layernorm.requires_grad_(False)
                layer.post_attention_layernorm.requires_grad_(False)

        return layer


# OverRep and the Table 4 ablations. Tuples are (attn block1, mlp block1, attn block2, mlp block2); a scalar applies to all four.
REPARAM_PRESETS = {
    'OverRep': dict(width_num_module=(1, 0, 1, 1), depth_num_module=(0, 1, 1, 1)),

    'AbBase': dict(width_num_module=0, depth_num_module=0),
    'AbAttnW': dict(width_num_module=(1, 0, 1, 0), depth_num_module=(0, 0, 0, 0)),
    'AbAttnD': dict(width_num_module=(0, 0, 0, 0), depth_num_module=(1, 0, 1, 0)),
    'AbMLPW': dict(width_num_module=(0, 1, 0, 1), depth_num_module=(0, 0, 0, 0)),
    'AbMLPD': dict(width_num_module=(0, 0, 0, 0), depth_num_module=(0, 1, 0, 1)),
    'AbW': dict(width_num_module=1, depth_num_module=0),
    'AbD': dict(width_num_module=0, depth_num_module=1),
    'AbWD': dict(width_num_module=1, depth_num_module=1),
    'AbHyb': dict(width_num_module=(1, 0, 1, 0), depth_num_module=(0, 1, 0, 1)),
}

MODEL_SPECS = {
    'llama2_7B': ('meta-llama/Llama-2-7b-hf', 'llama2-7B'),
    'llama2_13B': ('meta-llama/Llama-2-13b-hf', 'llama2-13B'),
    'llama3_3B': ('meta-llama/Llama-3.2-3B', 'llama3-3B'),
    'llama3_8B': ('meta-llama/Llama-3.1-8B', 'llama3-8B'),
    'qwen3_4B': ('Qwen/Qwen3-4B-Base', 'qwen3-4B'),
    'qwen3_8B': ('Qwen/Qwen3-8B-Base', 'qwen3-8B'),
    'qwen3_14B': ('Qwen/Qwen3-14B-Base', 'qwen3-14B'),
    'qwen3moe_30B_A3B': ('Qwen/Qwen3-30B-A3B-Base', 'qwen3moe-30B-A3B'),
}


def build_reparam_args(model_name, **kwargs):
    module_dict = get_specific_module(model_name)
    return [
        module_dict['decoder'],
        module_dict['rotary_embedding'],
        module_dict['norm'],
        module_dict['rep_mlp'],
        module_dict['rep_attn'],
        module_dict['rep_norm'],
        kwargs['target_layer'],
        kwargs['layer_interval'],
        kwargs['width_num_module'],
        kwargs['depth_num_module'],
        kwargs['va'],
        kwargs.get('rep_norm', False),
        1024,
    ]


def make_model_builder(hf_name, model_name, preset_name, preset, lm_cls=MultiDepthReparamLM):
    def builder(**kwargs):
        config = AutoConfig.from_pretrained(hf_name)
        full_kwargs = {**kwargs, **preset}
        return lm_cls(
            config,
            model_name,
            *build_reparam_args(model_name, **full_kwargs)
        )

    builder.__name__ = f"{model_name.replace('-', '_')}_{preset_name}"
    return register_model(builder)


for model_key, (hf_name, model_name) in MODEL_SPECS.items():
    _lm_cls = MoeMultiDepthReparamLM if 'moe' in model_key.lower() else MultiDepthReparamLM
    for preset_name, preset in REPARAM_PRESETS.items():
        fn_name = f"{model_key}_{preset_name}"
        globals()[fn_name] = make_model_builder(hf_name, model_name, preset_name, preset, _lm_cls)
