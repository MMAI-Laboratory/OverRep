import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRotaryEmbedding

from overrep.models.reparam_module import RepAttention, RepMLP, RepRMSNorm
from overrep.models.vanishing_activation import run_module_function


class LlamaRepRMSNorm(RepRMSNorm):
    pass


class LlamaRepMLP(RepMLP):
    pass


class LlamaRepAttention(RepAttention):
    pass


def check_mlp():
    config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B')
    dtype = torch.float32
    device = torch.device('cuda')

    hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
    depth_reparam_mlp = LlamaRepMLP(hidden_size, intermediate_size, 'silu', 3, 3, False)
    for p in depth_reparam_mlp.parameters():
        torch.nn.init.xavier_normal_(p)
    depth_reparam_mlp.eval()
    run_module_function(depth_reparam_mlp, total_steps=10, current_step=10)

    x = torch.rand(10, 32, hidden_size, dtype=dtype, device=device)

    org_mlp = LlamaMLP(config)
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
    config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-3B')
    print(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim)
    dtype = torch.float32
    device = torch.device("cuda")

    x = torch.rand(10, 1024, config.hidden_size, dtype=dtype, device=device)
    print('Input Shape: ', x.shape)
    rotary_emb = LlamaRotaryEmbedding(config)
    position_ids = torch.arange(0, 1024).unsqueeze(0).to(device)
    position_embeddings = rotary_emb(x, position_ids)

    attn = LlamaRepAttention(config.hidden_size, config.head_dim, config.num_attention_heads, config.num_key_value_heads,
                             3, 3, False)
    for p in attn.parameters():
        torch.nn.init.xavier_normal_(p)

    attn.eval()
    run_module_function(attn, total_steps=10, current_step=10)

    config._attn_implementation = 'sdpa'
    org_attn = LlamaAttention(config, 0)
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
    check_attn()
    check_mlp()
