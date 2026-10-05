from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from overrep.models.reparam_module import RepMLP, RepRMSNorm
from overrep.models.reparam_module_qwen import Qwen3RepAttention
from overrep.models.vanishing_activation import run_module_function


class Qwen3MoeRepRMSNorm(RepRMSNorm):
    pass


class Qwen3MoeRepAttention(Qwen3RepAttention):
    pass


class Qwen3MoeRepSparseMoeBlock(nn.Module):
    def __init__(self, config, width_num_module, depth_num_module, va):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [RepMLP(hidden_size, config.moe_intermediate_size, config.hidden_act,
                    width_num_module, depth_num_module, va) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def init_from_pretrained(self, gate_weight, expert_weights):
        self.gate.load_state_dict({'weight': gate_weight})
        self.gate.requires_grad_(False)

        assert len(expert_weights) == self.num_experts, \
            f"Expected {self.num_experts} expert weight triples, got {len(expert_weights)}"
        for expert, (gate_w, up_w, down_w) in zip(self.experts, expert_weights):
            expert.init_from_pretrained(gate_w, up_w, down_w)

    def freeze_pretrained(self):
        for expert in self.experts:
            expert.freeze_pretrained()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state = OrderedDict({'gate.weight': self.gate.weight})
        for i, expert in enumerate(self.experts):
            for k, v in expert.state_dict().items():
                state[f'experts.{i}.{k}'] = v
        return state


def _small_config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    return Qwen3MoeConfig(
        hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_hidden_layers=2, max_position_embeddings=256, vocab_size=128,
    )


def _randomize(module):
    for p in module.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.normal_(p, std=0.02)


def _report(name, out1, out2, atol=1e-4):
    diff = torch.abs(out1 - out2)
    ok = torch.allclose(out1, out2, atol=atol)
    print(f"[{name}] allclose={ok}  max_diff={diff.max().item():.3e}  mean_diff={diff.mean().item():.3e}")
    return ok


def check_moe_block():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = _small_config()

    hf_block = Qwen3MoeSparseMoeBlock(config)
    _randomize(hf_block)
    hf_block.eval().to(device)

    def hf_weights(block):
        gate_w = block.gate.weight.data
        experts = [(e.gate_proj.weight.data, e.up_proj.weight.data, e.down_proj.weight.data)
                   for e in block.experts]
        return gate_w, experts

    x = torch.rand(2, 16, config.hidden_size, device=device)

    rep = Qwen3MoeRepSparseMoeBlock(config, width_num_module=1, depth_num_module=1, va=True)
    rep.init_from_pretrained(*hf_weights(hf_block))
    run_module_function(rep, total_steps=10, current_step=10)
    rep.eval().to(device)
    with torch.no_grad():
        ok1 = _report('q3moe_block/identity-init', rep(x), hf_block(x)[0])

    _randomize(rep)
    run_module_function(rep, total_steps=10, current_step=10)
    merged = Qwen3MoeSparseMoeBlock(config)
    merged.load_state_dict(rep.state_dict())
    merged.eval().to(device)
    with torch.no_grad():
        ok2 = _report('q3moe_block/merged', rep(x), merged(x)[0])
    return ok1 and ok2


def check_full_layer():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

    from overrep.models.reparam_model import (
        CustomQwen3MoeDecoderLayer,
        MoeMultiDepthReparamLM,
        _moe_attn_weights,
        _moe_mlp_weights,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = _small_config()
    config._attn_implementation = 'sdpa'
    seq_len = 32

    hf_layer = Qwen3MoeDecoderLayer(config, 0)
    _randomize(hf_layer)
    hf_layer.eval().to(device)

    sd = {f'model.layers.0.{k}': v for k, v in hf_layer.state_dict().items()}

    rep_layer = MoeMultiDepthReparamLM._build_decoder_layer(
        MoeMultiDepthReparamLM,
        decoder_layer_module=CustomQwen3MoeDecoderLayer,
        attn_module=Qwen3MoeRepAttention,
        mlp_module=Qwen3MoeRepSparseMoeBlock,
        rep_norm_module=Qwen3MoeRepRMSNorm,
        input_layernorm_weights=sd['model.layers.0.input_layernorm.weight'],
        post_attention_layernorm_weights=sd['model.layers.0.post_attention_layernorm.weight'],
        attn_weights=_moe_attn_weights(sd, 0),
        mlp_weights=_moe_mlp_weights(sd, 0),
        config=config,
        target_layer=0,
        width_num_module=(1, 1),
        depth_num_module=(1, 1),
        rep_norm=True,
        va=True,
        freeze_pretrained=True,
    )
    rep_layer.eval().to(device)

    x = torch.rand(2, seq_len, config.hidden_size, device=device)
    rotary = Qwen3MoeRotaryEmbedding(config).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    def fwd_hf(layer):
        out = layer(x, position_ids=position_ids, position_embeddings=position_embeddings)
        return out[0] if isinstance(out, tuple) else out

    def fwd_rep():
        return rep_layer(x, attention_mask=None, position_ids=position_ids,
                         position_embeddings=position_embeddings)

    run_module_function(rep_layer, total_steps=10, current_step=10)
    with torch.no_grad():
        ok1 = _report('q3moe_layer/identity-init', fwd_rep(), fwd_hf(hf_layer))

    _randomize(rep_layer)
    run_module_function(rep_layer, total_steps=10, current_step=10)
    merged_sd = {}
    for mod_name in ['input_layernorm', 'post_attention_layernorm', 'mlp', 'self_attn']:
        for k, v in getattr(rep_layer, mod_name).state_dict().items():
            merged_sd[f'{mod_name}.{k}'] = v
    fresh = Qwen3MoeDecoderLayer(config, 0)
    fresh.load_state_dict(merged_sd)
    fresh.eval().to(device)
    with torch.no_grad():
        ok2 = _report('q3moe_layer/merged', fwd_rep(), fwd_hf(fresh))

    frozen = [n for n, p in rep_layer.named_parameters() if not p.requires_grad]
    assert any('mlp.gate.weight' in n for n in frozen), "router must be frozen"
    assert any('linears.0' in n for n in frozen), "pretrained branches must be frozen"
    print('[q3moe_layer/freeze] router + pretrained branches frozen: OK')

    return ok1 and ok2


if __name__ == '__main__':
    results = {
        'moe_block': check_moe_block(),
        'full_layer': check_full_layer(),
    }
    print('---')
    failed = [k for k, v in results.items() if not v]
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print('ALL CHECKS PASSED')
