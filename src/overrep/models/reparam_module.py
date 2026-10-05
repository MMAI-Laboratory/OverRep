from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.integrations import use_kernel_forward_from_hub
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from overrep.models.vanishing_activation import VanishingActivation


@use_kernel_forward_from_hub("RMSNorm")
class RepRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight_base = nn.Parameter(torch.ones(hidden_size))

        self.weight_scale = nn.Parameter(torch.zeros(hidden_size))
        self.weight_bias = nn.Parameter(torch.zeros(hidden_size))

        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        eff_weight = self.weight_base * (1.0 + self.weight_scale) + self.weight_bias
        return eff_weight * hidden_states.to(input_dtype)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {'weight': self.weight_base * (1.0 + self.weight_scale) + self.weight_bias}

    def init_from_pretrained(self, weight):
        self.weight_base.data.copy_(weight)

    def freeze_pretrained(self):
        self.weight_base.requires_grad_(False)


class WidthRepLinear(nn.Module):
    def __init__(self, in_features, out_features, num_module):
        super().__init__()
        num_module = num_module + 1
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for _ in range(num_module)])
        self.num_module = num_module

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = 0
        for linear in self.linears:
            out = out + linear(x)
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        weight = 0
        for linear in self.linears:
            weight = weight + linear.weight
        return {'weight': weight}

    def init_from_pretrained(self, weight):
        self.linears[0].load_state_dict({'weight': weight})
        for i in range(1, self.num_module):
            torch.nn.init.zeros_(self.linears[i].weight)

    def freeze_pretrained(self):
        self.linears[0].requires_grad_(False)


class DepthRepLinear(nn.Module):
    def __init__(self, in_features, out_features, num_module, va=False, pre_act=False):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for _ in range(num_module)])
        self.num_module = num_module
        self.pre_act = pre_act
        if va:
            self.act = VanishingActivation()
        else:
            self.act = nn.Identity()

        for i in range(self.num_module):
            torch.nn.init.eye_(self.linears[i].weight)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for linear in self.linears:
            if self.pre_act:
                x = linear(self.act(x))
            else:
                x = self.act(linear(x))
        return x

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        dtype = self.linears[0].weight.dtype

        weight = self.linears[-1].weight.data.float()
        for i in reversed(range(self.num_module - 1)):
            weight = torch.matmul(weight, self.linears[i].weight.data.float())
        return {'weight': weight.to(dtype)}


class DownReparamLinear(nn.Module):
    def __init__(self, in_features, out_features, width_num_module, depth_num_module, va):
        super().__init__()
        self.proj1 = WidthRepLinear(in_features, out_features, width_num_module)
        if depth_num_module == 0:
            self.proj2 = nn.Identity()
        else:
            self.proj2 = DepthRepLinear(out_features, out_features, depth_num_module, va, pre_act=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj2(self.proj1(x))
        return x

    def init_from_pretrained(self, weight):
        self.proj1.init_from_pretrained(weight)

    def freeze_pretrained(self):
        self.proj1.freeze_pretrained()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if isinstance(self.proj2, nn.Identity):
            return self.proj1.state_dict()

        dtype = self.proj1.linears[0].weight.dtype
        weight = torch.matmul(self.proj2.state_dict()['weight'].float(),
                              self.proj1.state_dict()['weight'].float())

        return {'weight': weight.to(dtype)}


class UpReparamLinear(nn.Module):
    def __init__(self, in_features, out_features, width_num_module, depth_num_module, va):
        super().__init__()
        if depth_num_module == 0:
            self.proj1 = nn.Identity()
        else:
            self.proj1 = DepthRepLinear(in_features, in_features, depth_num_module, va)
        self.proj2 = WidthRepLinear(in_features, out_features, width_num_module)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj2(self.proj1(x))
        return x

    def init_from_pretrained(self, weight):
        self.proj2.init_from_pretrained(weight)

    def freeze_pretrained(self):
        self.proj2.freeze_pretrained()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if isinstance(self.proj1, nn.Identity):
            return self.proj2.state_dict()

        dtype = self.proj2.linears[0].weight.dtype
        weight = torch.matmul(self.proj2.state_dict()['weight'].float(),
                              self.proj1.state_dict()['weight'].float())
        return {'weight': weight.to(dtype)}


class RepMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, width_num_module, depth_num_module, va):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = DownReparamLinear(hidden_size, intermediate_size, width_num_module, depth_num_module, va)
        self.up_proj = DownReparamLinear(hidden_size, intermediate_size, width_num_module, depth_num_module, va)
        self.down_proj = UpReparamLinear(intermediate_size, hidden_size, width_num_module, depth_num_module, va)

        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def init_from_pretrained(self, gate_weights, up_weights, down_weights):
        self.gate_proj.init_from_pretrained(gate_weights)
        self.up_proj.init_from_pretrained(up_weights)
        self.down_proj.init_from_pretrained(down_weights)

    def freeze_pretrained(self):
        self.gate_proj.freeze_pretrained()
        self.up_proj.freeze_pretrained()
        self.down_proj.freeze_pretrained()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        gate_proj_weight = self.gate_proj.state_dict()['weight']
        up_proj_weight = self.up_proj.state_dict()['weight']
        down_proj_weight = self.down_proj.state_dict()['weight']

        return OrderedDict({'gate_proj.weight': gate_proj_weight, 'up_proj.weight': up_proj_weight,
                            'down_proj.weight': down_proj_weight})


class RepAttention(nn.Module):
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
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def init_from_pretrained(self, q_weights, k_weights, v_weights, o_weights):
        self.q_proj.init_from_pretrained(q_weights)
        self.k_proj.init_from_pretrained(k_weights)
        self.v_proj.init_from_pretrained(v_weights)
        self.o_proj.init_from_pretrained(o_weights)

    def freeze_pretrained(self):
        self.q_proj.freeze_pretrained()
        self.k_proj.freeze_pretrained()
        self.v_proj.freeze_pretrained()
        self.o_proj.freeze_pretrained()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        q_proj_weight = self.q_proj.state_dict()['weight']
        k_proj_weight = self.k_proj.state_dict()['weight']
        v_proj_weight = self.v_proj.state_dict()['weight']
        o_proj_weight = self.o_proj.state_dict()['weight']

        return OrderedDict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight,
                            'v_proj.weight': v_proj_weight, 'o_proj.weight': o_proj_weight})

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None,
                **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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


def check_depth_rep_linear():
    hidden_size = 1024
    device = torch.device('cuda')
    dtype = torch.float32
    x = torch.rand(10, 1024, 1024, dtype=dtype, device=device)
    rep_linear = DepthRepLinear(hidden_size, hidden_size, 3)
    rep_linear.to(device).to(dtype)

    reparam_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    reparam_linear.load_state_dict(rep_linear.state_dict())
    reparam_linear.to(device).to(dtype)

    with torch.no_grad():
        out1 = rep_linear(x)
        out2 = reparam_linear(x)
    print(torch.allclose(out1, out2, atol=1e-3), torch.abs(out1 - out2).mean())


def check_width_rep_linear():
    hidden_size = 1024
    device = torch.device('cuda')
    dtype = torch.float32
    x = torch.rand(10, 1024, 1024, dtype=dtype, device=device)
    rep_linear = WidthRepLinear(hidden_size, hidden_size, 3)
    rep_linear.to(device).to(dtype)

    reparam_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    reparam_linear.load_state_dict(rep_linear.state_dict())
    reparam_linear.to(device).to(dtype)

    with torch.no_grad():
        out1 = rep_linear(x)
        out2 = reparam_linear(x)
    print(torch.allclose(out1, out2, atol=1e-3), torch.abs(out1 - out2).mean())


if __name__ == '__main__':
    check_depth_rep_linear()
    check_width_rep_linear()
