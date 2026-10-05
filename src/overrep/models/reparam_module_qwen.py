from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from overrep.models.reparam_module import DownReparamLinear, RepMLP, RepRMSNorm, UpReparamLinear
from overrep.models.vanishing_activation import run_module_function


class Qwen3RepRMSNorm(RepRMSNorm):
    pass


class Qwen3RepMLP(RepMLP):
    pass


class Qwen3RepAttention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_attention_heads, num_key_value_heads, width_num_module,
                 depth_num_module, va, attention_dropout=0.):
        super().__init__()
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.attention_dropout = attention_dropout
        self.attention_interface = sdpa_attention_forward
        self.config = SimpleNamespace(_attn_implementation="sdpa")

        self.q_proj = DownReparamLinear(hidden_size, num_attention_heads * self.head_dim, width_num_module,
                                        depth_num_module, va)
        self.k_proj = DownReparamLinear(hidden_size, num_key_value_heads * self.head_dim, width_num_module,
                                        depth_num_module, va)
        self.v_proj = DownReparamLinear(hidden_size, num_key_value_heads * self.head_dim, width_num_module,
                                        depth_num_module, va)
        self.o_proj = UpReparamLinear(num_attention_heads * self.head_dim, hidden_size, width_num_module,
                                      depth_num_module, va)
        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def init_from_pretrained(self, q_weights, k_weights, v_weights, o_weights, qn_weights, kn_weights):
        self.q_proj.init_from_pretrained(q_weights)
        self.k_proj.init_from_pretrained(k_weights)
        self.v_proj.init_from_pretrained(v_weights)
        self.o_proj.init_from_pretrained(o_weights)
        self.q_norm.load_state_dict({'weight': qn_weights})
        self.k_norm.load_state_dict({'weight': kn_weights})

    def freeze_pretrained(self):
        self.q_proj.freeze_pretrained()
        self.k_proj.freeze_pretrained()
        self.v_proj.freeze_pretrained()
        self.o_proj.freeze_pretrained()
        self.q_norm.weight.requires_grad_(False)
        self.k_norm.weight.requires_grad_(False)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        q_proj_weight = self.q_proj.state_dict()['weight']
        k_proj_weight = self.k_proj.state_dict()['weight']
        v_proj_weight = self.v_proj.state_dict()['weight']
        o_proj_weight = self.o_proj.state_dict()['weight']
        qn_weight = self.q_norm.state_dict()['weight']
        kn_weight = self.k_norm.state_dict()['weight']

        return OrderedDict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight,
                            'v_proj.weight': v_proj_weight, 'o_proj.weight': o_proj_weight,
                            'q_norm.weight': qn_weight, 'k_norm.weight': kn_weight})

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None,
                **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous().to(dtype=hidden_states.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def check_mlp():
    config = AutoConfig.from_pretrained('Qwen/Qwen3-4B-Base')
    dtype = torch.float32
    device = torch.device('cuda')

    hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
    depth_reparam_mlp = Qwen3RepMLP(hidden_size, intermediate_size, 'silu', 1, 1, False)
    for p in depth_reparam_mlp.parameters():
        torch.nn.init.xavier_normal_(p)
    depth_reparam_mlp.eval()
    run_module_function(depth_reparam_mlp, total_steps=10, current_step=10)

    x = torch.rand(10, 32, hidden_size, dtype=dtype, device=device)

    org_mlp = Qwen3MLP(config)
    org_mlp.eval()

    reparam_state_dict = depth_reparam_mlp.state_dict()
    print(reparam_state_dict.keys())
    org_mlp.load_state_dict(reparam_state_dict)

    org_mlp.to(dtype).to(device)
    depth_reparam_mlp.to(dtype).to(device)
    with torch.no_grad():
        out1 = depth_reparam_mlp(x)
        out2 = org_mlp(x)

    print(torch.allclose(out1, out2, atol=1e-3), torch.abs(out1 - out2).mean())


def check_attn():
    config = AutoConfig.from_pretrained('Qwen/Qwen3-4B-Base')
    dtype = torch.float32
    device = torch.device("cuda")

    x = torch.rand(10, 1024, config.hidden_size, dtype=dtype, device=device)
    print('Input Shape: ', x.shape)
    rotary_emb = Qwen3RotaryEmbedding(config)
    position_ids = torch.arange(0, 1024).unsqueeze(0).to(device)
    position_embeddings = rotary_emb(x, position_ids)

    attn = Qwen3RepAttention(config.hidden_size, config.head_dim, config.num_attention_heads,
                             config.num_key_value_heads,
                             1, 1, False)

    attn.eval()
    run_module_function(attn, total_steps=10, current_step=10)

    config._attn_implementation = 'sdpa'
    org_attn = Qwen3Attention(config, 0)
    org_attn.eval()

    reparam_state_dict = attn.state_dict()
    print(reparam_state_dict.keys())
    org_attn.load_state_dict(reparam_state_dict)

    attn.to(dtype).to(device)
    org_attn.to(dtype).to(device)
    with torch.no_grad():
        out = attn(x, position_embeddings, None)
        org_out = org_attn(x, position_embeddings, None)

    print(torch.allclose(out[0], org_out[0], atol=1e-3), torch.abs(out[0] - org_out[0]).mean())


if __name__ == '__main__':
    check_mlp()
    check_attn()
