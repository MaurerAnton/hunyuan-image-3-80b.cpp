#!/usr/bin/env python3
"""
Full model reference: embedding + N layers + final norm + lm_head
Dumps intermediate values for C++ comparison.

Usage:
  python scripts/full_ref.py --shard-dir weights --seq-len 4 --outdir test_full
"""

import argparse, json, struct, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import safetensors.torch

def load_all_shards(shard_dir, index_path):
    """Load all tensors from all shards into one big dict."""
    with open(index_path) as f:
        idx = json.load(f)
    
    all_tensors = {}
    shard_files = set(idx['weight_map'].values())
    shard_dir = Path(shard_dir)
    
    for sf_name in sorted(shard_files):
        sf_path = shard_dir / sf_name
        if not sf_path.exists():
            print(f"  SKIP: {sf_path}")
            continue
        shard = safetensors.torch.load_file(str(sf_path))
        for k, v in shard.items():
            all_tensors[k] = v.float()
        print(f"  Loaded {sf_name}: {len(shard)} tensors")
    
    return all_tensors


class FullModelRef:
    def __init__(self, shard_dir, index_path, num_layers=1):
        self.all_tensors = load_all_shards(shard_dir, index_path)
        
        # Config
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path) as f:
            self.cfg = json.load(f)
        
        self.hidden_size = self.cfg['hidden_size']        # 4096
        self.num_heads = self.cfg['num_attention_heads']   # 32
        self.num_kv_heads = self.cfg['num_key_value_heads'] # 8
        self.head_dim = self.cfg['attention_head_dim']     # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rms_norm_eps = self.cfg['rms_norm_eps']
        self.rope_theta = self.cfg['rope_theta']
        self.num_experts = self.cfg['num_experts']
        self.moe_topk = self.cfg['moe_topk'][0]
        self.moe_intermediate = self.cfg['moe_intermediate_size'][0]
        self.vocab_size = self.cfg['vocab_size']
        self.num_layers = num_layers
        
        # Global weights
        self.wte = self.all_tensors['model.wte.weight']  # [vocab, hidden]
        self.ln_f = self.all_tensors['model.ln_f.weight'] # [hidden]
        self.lm_head = self.all_tensors['lm_head.weight'] # [vocab, hidden]
        
        print(f"wte: {list(self.wte.shape)}")
        print(f"ln_f: {list(self.ln_f.shape)}")
        print(f"lm_head: {list(self.lm_head.shape)}")
    
    def rms_norm(self, x, weight):
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        return (xf * rms * weight.float()).to(x.dtype)
    
    def apply_rope(self, q, k, positions):
        bsz, n_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape
        device = q.device
        
        if positions is not None:
            pos = positions.float()
        else:
            pos = torch.arange(seq_len, device=device).float()
        
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        freqs = torch.outer(pos, theta)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        
        def rotate(x, n_h):
            xf = x.float()
            x1, x2 = xf[..., ::2], xf[..., 1::2]
            r = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            out = torch.empty_like(r)
            out[..., ::2] = r[..., :head_dim//2]
            out[..., 1::2] = r[..., head_dim//2:]
            return out.to(x.dtype)
        
        q_out = rotate(q, n_heads)
        k_out = rotate(k, n_kv_heads)
        return q_out, k_out
    
    def attention(self, hidden_states, layer_idx, positions=None):
        bsz, seq_len, _ = hidden_states.shape
        prefix = f'model.layers.{layer_idx}.'
        
        qkv_w = self.all_tensors[f'{prefix}self_attn.qkv_proj.weight']
        o_w = self.all_tensors[f'{prefix}self_attn.o_proj.weight']
        q_norm_w = self.all_tensors[f'{prefix}self_attn.query_layernorm.weight']
        k_norm_w = self.all_tensors[f'{prefix}self_attn.key_layernorm.weight']
        
        hidden_q = self.num_heads * self.head_dim   # 4096
        hidden_kv = self.num_kv_heads * self.head_dim # 1024
        
        qkv = F.linear(hidden_states, qkv_w)
        q = qkv[..., :hidden_q]
        k = qkv[..., hidden_q:hidden_q + hidden_kv]
        v = qkv[..., hidden_q + hidden_kv:]
        
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.apply_rope(q, k, positions)
        
        q = self.rms_norm(q.transpose(1, 2).reshape(-1, self.head_dim), q_norm_w)
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.rms_norm(k.transpose(1, 2).reshape(-1, self.head_dim), k_norm_w)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return F.linear(attn_out, o_w)
    
    def swiglu_ffn(self, x, gate_up_w, down_w):
        gate_up = F.linear(x, gate_up_w)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(gate * F.silu(up), down_w)
    
    def moe(self, hidden_states, layer_idx):
        bsz, seq_len, hidden_size = hidden_states.shape
        prefix = f'model.layers.{layer_idx}.'
        flat = hidden_states.reshape(-1, hidden_size)
        
        # Shared expert
        shared_gate_up = self.all_tensors[f'{prefix}mlp.shared_mlp.gate_and_up_proj.weight']
        shared_down = self.all_tensors[f'{prefix}mlp.shared_mlp.down_proj.weight']
        shared_out = self.swiglu_ffn(flat, shared_gate_up, shared_down)
        
        # Router
        router_w = self.all_tensors[f'{prefix}mlp.gate.wg.weight']
        router_logits = F.linear(flat, router_w)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, self.moe_topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Routed experts
        combined = torch.zeros_like(flat)
        for token_idx in range(flat.shape[0]):
            token_vec = flat[token_idx:token_idx+1]
            for k_idx in range(self.moe_topk):
                expert_idx = topk_indices[token_idx, k_idx].item()
                weight = topk_weights[token_idx, k_idx]
                gate_up = self.all_tensors[f'{prefix}mlp.experts.{expert_idx}.gate_and_up_proj.weight']
                down = self.all_tensors[f'{prefix}mlp.experts.{expert_idx}.down_proj.weight']
                expert_out = self.swiglu_ffn(token_vec, gate_up, down)
                combined[token_idx] += weight * expert_out[0]
        
        output = (shared_out + combined).reshape(bsz, seq_len, hidden_size)
        return output
    
    def decoder_layer(self, hidden_states, layer_idx, positions=None):
        prefix = f'model.layers.{layer_idx}.'
        input_ln = self.all_tensors[f'{prefix}input_layernorm.weight']
        post_ln = self.all_tensors[f'{prefix}post_attention_layernorm.weight']
        
        residual = hidden_states
        normed = self.rms_norm(hidden_states, input_ln)
        attn_out = self.attention(normed, layer_idx, positions)
        hidden_states = residual + attn_out
        
        residual = hidden_states
        normed = self.rms_norm(hidden_states, post_ln)
        moe_out = self.moe(normed, layer_idx)
        hidden_states = residual + moe_out
        
        return hidden_states
    
    def forward(self, input_ids, positions=None):
        bsz, seq_len = input_ids.shape
        h = F.embedding(input_ids, self.wte)  # [batch, seq, hidden]
        
        debug_hidden = [h.clone()]
        
        for layer in range(self.num_layers):
            h = self.decoder_layer(h, layer, positions)
            debug_hidden.append(h.clone())
        
        # Final norm
        h_normed = self.rms_norm(h, self.ln_f)
        
        # LM head
        logits = F.linear(h_normed, self.lm_head)
        
        return logits, debug_hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-dir', required=True)
    parser.add_argument('--index', default=None, help='index.json path')
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--outdir', default='test_full')
    args = parser.parse_args()
    
    shard_dir = args.shard_dir
    index_path = args.index or os.path.join(os.path.dirname(shard_dir), 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(os.path.dirname(shard_dir), 'model_index.json')
    
    print(f"Index: {index_path}")
    ref = FullModelRef(shard_dir, index_path, args.layers)
    
    torch.manual_seed(42)
    input_ids = torch.randint(0, min(1000, ref.vocab_size), (1, args.seq_len))
    
    with torch.no_grad():
        logits, debug_hidden = ref.forward(input_ids)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Dump
    input_ids.numpy().astype(np.int32).tofile(str(outdir / 'input_ids.bin'))
    logits.float().numpy().tofile(str(outdir / 'logits.bin'))
    
    for i, h in enumerate(debug_hidden):
        h.float().numpy().tofile(str(outdir / f'hidden_{i}.bin'))
    
    # RMS
    def rms(t): return float(t.float().pow(2).mean().sqrt())
    print("\nRMS values:")
    print(f"  input_ids: {input_ids.tolist()}")
    print(f"  embedding: RMS={rms(debug_hidden[0]):.6f}")
    for i in range(1, len(debug_hidden)):
        print(f"  after layer {i-1}: RMS={rms(debug_hidden[i]):.6f}")
    print(f"  logits: RMS={rms(logits):.6f}")
    
    # Save shapes
    shapes = {
        'input_ids': list(input_ids.shape),
        'logits': list(logits.shape),
    }
    for i, h in enumerate(debug_hidden):
        shapes[f'hidden_{i}'] = list(h.shape)
    with open(outdir / 'shapes.json', 'w') as f:
        json.dump(shapes, f, indent=2)
    
    print(f"\nDumped to {outdir}/")


if __name__ == '__main__':
    main()
