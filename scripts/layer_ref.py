#!/usr/bin/env python3
"""
Layer-by-layer reference for Hunyuan Image 3.0 transformer blocks.
Loads model weights, runs a single decoder layer, dumps intermediate tensors
for C++ comparison.

Usage:
  python scripts/layer_ref.py --layer 0 --shard-dir /path/to/safetensors/

Output:
  layer_0_input.bin      - input to layer 0 (hidden_states)
  layer_0_output.bin     - output of layer 0
  layer_0_attn_q.bin     - query after QKV proj + RoPE + QK norm
  layer_0_attn_k.bin     - key after QKV proj + RoPE + QK norm
  layer_0_attn_v.bin     - value after QKV proj
  layer_0_attn_out.bin   - attention output (after o_proj, before residual)
  layer_0_moe_out.bin    - MoE output (shared + routed, before residual)
  layer_0_router_logits.bin - router logits
  layer_0_router_weights.bin - routing weights (top-8)
  layer_0_router_indices.bin - expert indices (top-8)
  layer_0_rms.txt        - RMS values at each stage
"""

import argparse
import json
import os
import struct
import sys
import numpy as np
from pathlib import Path

# We don't import torch to keep it light for now.
# The user can install torch+hunyuan deps later.
# For the reference, we do a numpy-level simulation.

import torch
import torch.nn.functional as F

# Add the hunyuan.cpp dir to path to import the modeling code
sys.path.insert(0, str(Path(__file__).parent.parent))

# We need huggingface transformers
from transformers import AutoConfig

# Monkey-patch so we don't need the full HF env
import importlib.util

def load_safetensors_shard(path):
    """Load a single safetensors shard into torch tensors dict."""
    import safetensors.torch
    return safetensors.torch.load_file(path)

def load_model_config(config_path):
    with open(config_path) as f:
        return json.load(f)


class LayerRef:
    """Reference implementation of ONE Hunyuan decoder layer in numpy/PyTorch."""
    
    def __init__(self, layer_idx, shard_dir):
        self.layer_idx = layer_idx
        self.shard_dir = Path(shard_dir)
        self.config = load_model_config(str(Path(__file__).parent.parent / 'config.json'))
        
        # Architecture params
        self.hidden_size = self.config['hidden_size']       # 4096
        self.num_heads = self.config['num_attention_heads']  # 32
        self.num_kv_heads = self.config['num_key_value_heads'] # 8
        self.head_dim = self.config['attention_head_dim']    # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 4
        self.rms_norm_eps = self.config['rms_norm_eps']
        self.rope_theta = self.config['rope_theta']
        self.num_experts = self.config['num_experts']  # 64
        self.moe_topk = self.config['moe_topk'][layer_idx]  # 8
        self.moe_intermediate = self.config['moe_intermediate_size'][layer_idx]  # 3072
        
        # Find which shard contains this layer
        self._find_shard()
        self._load_weights()
    
    def _find_shard(self):
        """Find ALL safetensors shards that have this layer's weights."""
        # Look for index.json
        index_path = self.shard_dir / 'model.safetensors.index.json'
        if not index_path.exists():
            index_path = self.shard_dir.parent / 'model.safetensors.index.json'
        if not index_path.exists():
            index_path = self.shard_dir.parent / 'model_index.json'
        
        self.shard_files = []
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index['weight_map']
            # Find all shards for this layer
            seen = set()
            for key, shard_name in weight_map.items():
                if f'model.layers.{self.layer_idx}.' in key:
                    if shard_name not in seen:
                        seen.add(shard_name)
                        self.shard_files.append(self.shard_dir / shard_name)
        else:
            self.shard_files = [self.shard_dir / 'model.safetensors']
        
        print(f"Layer {self.layer_idx} weights in {len(self.shard_files)} shards:")
        for sf in self.shard_files:
            print(f"  {sf}")
    
    def _load_weights(self):
        """Load weights for this layer from safetensors (across multiple shards)."""
        # Load all shards into a combined dict
        all_tensors = {}
        for sf_path in self.shard_files:
            if not sf_path.exists():
                print(f"  SKIP missing shard: {sf_path}")
                continue
            shard = load_safetensors_shard(str(sf_path))
            for k, v in shard.items():
                if k.startswith(f'model.layers.{self.layer_idx}.'):
                    all_tensors[k] = v
            print(f"  Loaded {len(shard)} tensors from {sf_path.name}, kept {len(all_tensors)} layer tensors")
        
        prefix = f'model.layers.{self.layer_idx}.'
        
        # RMSNorm weights
        self.input_ln_weight = all_tensors[f'{prefix}input_layernorm.weight']
        self.post_attn_ln_weight = all_tensors[f'{prefix}post_attention_layernorm.weight']
        
        # Attention weights
        qkv_weight_raw = all_tensors[f'{prefix}self_attn.qkv_proj.weight']
        self.qkv_weight = qkv_weight_raw
        self.qkv_bias = None
        
        o_weight_raw = all_tensors[f'{prefix}self_attn.o_proj.weight']
        self.o_weight = o_weight_raw
        
        # QK norm weights
        self.q_norm_weight = all_tensors[f'{prefix}self_attn.query_layernorm.weight']
        self.k_norm_weight = all_tensors[f'{prefix}self_attn.key_layernorm.weight']
        
        # MoE: shared expert
        self.shared_gate_up = all_tensors[f'{prefix}mlp.shared_mlp.gate_and_up_proj.weight']
        self.shared_down = all_tensors[f'{prefix}mlp.shared_mlp.down_proj.weight']
        
        # MoE: router
        self.router_weight = all_tensors[f'{prefix}mlp.gate.wg.weight']
        
        # MoE: 64 routed experts
        self.expert_gate_up = []
        self.expert_down = []
        for e in range(self.num_experts):
            gu = all_tensors[f'{prefix}mlp.experts.{e}.gate_and_up_proj.weight']
            dn = all_tensors[f'{prefix}mlp.experts.{e}.down_proj.weight']
            self.expert_gate_up.append(gu)
            self.expert_down.append(dn)
        
        print(f"Loaded {len(self.expert_gate_up)} experts for layer {self.layer_idx}")
        print(f"  QKV weight: {list(self.qkv_weight.shape)}")
        print(f"  O weight:   {list(self.o_weight.shape)}")
        print(f"  Expert gate_up: {list(self.expert_gate_up[0].shape)}")
        print(f"  Expert down:    {list(self.expert_down[0].shape)}")
        
        # Convert all weights to float32 for computation
        self.input_ln_weight = self.input_ln_weight.float()
        self.post_attn_ln_weight = self.post_attn_ln_weight.float()
        self.qkv_weight = self.qkv_weight.float()
        self.o_weight = self.o_weight.float()
        self.q_norm_weight = self.q_norm_weight.float()
        self.k_norm_weight = self.k_norm_weight.float()
        self.shared_gate_up = self.shared_gate_up.float()
        self.shared_down = self.shared_down.float()
        self.router_weight = self.router_weight.float()
        self.expert_gate_up = [w.float() for w in self.expert_gate_up]
        self.expert_down = [w.float() for w in self.expert_down]
    
    def rms_norm(self, x, weight):
        """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        return (x_float * rms * weight.float()).to(x.dtype)
    
    def apply_2d_rope(self, q, k, positions):
        """
        Apply 2D RoPE to queries and keys.
        Hunyuan uses 2D rotary embeddings with separate x/y frequency bands.
        
        q, k: [batch, heads, seq_len, head_dim]
        positions: tuple of (x_pos, y_pos) each [batch, seq_len]
        """
        # Simplified: for generation, positions are sequential 1D
        # Full 2D RoPE is more complex. For layer verification,
        # we can test with 1D positions first.
        bsz, n_heads, seq_len, head_dim = q.shape
        
        # Standard 1D RoPE
        device = q.device
        pos = torch.arange(seq_len, device=device).float()
        if positions is not None:
            pos = positions.float()
        
        # Compute frequencies
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        freqs = torch.outer(pos, theta)  # [seq_len, head_dim/2]
        
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_rot = q.float()
        k_rot = k.float()
        
        q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
        k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]
        
        q_rot = torch.cat([
            q1 * cos - q2 * sin,
            q1 * sin + q2 * cos
        ], dim=-1)
        k_rot = torch.cat([
            k1 * cos - k2 * sin,
            k1 * sin + k2 * cos
        ], dim=-1)
        
        # Interleave back to original order
        q_out = torch.empty_like(q_rot)
        q_out[..., ::2] = q_rot[..., :head_dim//2]
        q_out[..., 1::2] = q_rot[..., head_dim//2:]
        
        k_out = torch.empty_like(k_rot)
        k_out[..., ::2] = k_rot[..., :head_dim//2]
        k_out[..., 1::2] = k_rot[..., head_dim//2:]
        
        return q_out.to(q.dtype), k_out.to(k.dtype)
    
    def attention_forward(self, hidden_states, positions=None):
        """Full attention forward pass."""
        bsz, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        
        # Split into Q, K, V
        # QKV layout: [Q (4096), K (1024), V (1024)]
        hidden_q = self.num_heads * self.head_dim  # 4096
        hidden_kv = self.num_kv_heads * self.head_dim  # 1024
        
        q = qkv[..., :hidden_q]
        k = qkv[..., hidden_q:hidden_q + hidden_kv]
        v = qkv[..., hidden_q + hidden_kv:]
        
        # Reshape to [batch, heads, seq, head_dim]
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.apply_2d_rope(q, k, positions)
        
        # QK Normalization
        q = self.rms_norm(q.transpose(1, 2).reshape(-1, self.head_dim), self.q_norm_weight)
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.rms_norm(k.transpose(1, 2).reshape(-1, self.head_dim), self.k_norm_weight)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Save Q, K, V for C++ comparison
        self._debug_q = q.clone()
        self._debug_k = k.clone()
        self._debug_v = v.clone()
        
        # GQA: repeat KV to match Q heads
        k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
        k = k.reshape(bsz, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
        v = v.reshape(bsz, self.num_heads, seq_len, self.head_dim)
        
        # SDPA
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        
        # Output projection
        out = F.linear(attn_out, self.o_weight)
        
        return out
    
    def moe_forward(self, hidden_states):
        """MoE forward: shared expert + top-k routed experts."""
        bsz, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)  # [bsz*seq_len, hidden]
        
        # Shared expert
        shared_out = self._swiglu_ffn(flat, self.shared_gate_up, self.shared_down)
        
        # Router
        router_logits = F.linear(flat.float(), self.router_weight.float())  # [N, 64]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(router_probs, self.moe_topk, dim=-1)
        
        # Normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Save router outputs for C++ comparison
        self._debug_router_logits = router_logits.clone()
        self._debug_router_weights = topk_weights.clone()
        self._debug_router_indices = topk_indices.clone()
        
        # Compute routed expert outputs efficiently
        # For each token, run only its top-k experts
        combined = torch.zeros_like(flat)
        
        for token_idx in range(flat.shape[0]):
            token_vec = flat[token_idx:token_idx+1]  # [1, hidden]
            for k_idx in range(self.moe_topk):
                expert_idx = topk_indices[token_idx, k_idx].item()
                weight = topk_weights[token_idx, k_idx]
                expert_out = self._swiglu_ffn(token_vec, 
                                              self.expert_gate_up[expert_idx],
                                              self.expert_down[expert_idx])
                combined[token_idx] += weight * expert_out[0]
        
        # Shared + routed
        output = (shared_out + combined).reshape(bsz, seq_len, hidden_size)
        return output
    
    def _swiglu_ffn(self, x, gate_up_weight, down_weight):
        """SwiGLU FFN: down(silu(gate(x)) * up(x))"""
        # gate_up_weight is [2*intermediate, hidden] = [6144, 4096]
        gate_up = F.linear(x, gate_up_weight)  # [N, 6144]
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(gate * F.silu(up), down_weight)  # [N, 4096]
    
    def forward(self, hidden_states, positions=None):
        """
        Full decoder layer forward.
        Returns output hidden states.
        Also stores debug tensors for C++ comparison.
        """
        self._debug_input = hidden_states.clone()
        
        # input_layernorm → attention → residual
        residual = hidden_states
        normed = self.rms_norm(hidden_states, self.input_ln_weight)
        attn_out = self.attention_forward(normed, positions)
        self._debug_attn_out = attn_out.clone()
        hidden_states = residual + attn_out
        
        # post_attention_layernorm → MoE → residual
        residual = hidden_states
        normed = self.rms_norm(hidden_states, self.post_attn_ln_weight)
        moe_out = self.moe_forward(normed)
        self._debug_moe_out = moe_out.clone()
        hidden_states = residual + moe_out
        
        self._debug_output = hidden_states.clone()
        
        return hidden_states
    
    def dump_tensors(self, outdir):
        """Dump all intermediate tensors as raw float32 binaries."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        prefix = f'layer_{self.layer_idx}'
        
        def save_bin(name, t):
            t.float().numpy().tofile(str(outdir / f'{prefix}_{name}.bin'))
        
        save_bin('input', self._debug_input)
        save_bin('output', self._debug_output)
        save_bin('attn_q', self._debug_q)
        save_bin('attn_k', self._debug_k)
        save_bin('attn_v', self._debug_v)
        save_bin('attn_out', self._debug_attn_out)
        save_bin('moe_out', self._debug_moe_out)
        save_bin('router_logits', self._debug_router_logits)
        save_bin('router_weights', self._debug_router_weights)
        save_bin('router_indices', self._debug_router_indices)
        
        # Also save shapes
        shapes = {
            'input': list(self._debug_input.shape),
            'output': list(self._debug_output.shape),
            'attn_q': list(self._debug_q.shape),
            'attn_k': list(self._debug_k.shape),
            'attn_v': list(self._debug_v.shape),
            'attn_out': list(self._debug_attn_out.shape),
            'moe_out': list(self._debug_moe_out.shape),
            'router_logits': list(self._debug_router_logits.shape),
            'router_weights': list(self._debug_router_weights.shape),
            'router_indices': list(self._debug_router_indices.shape),
        }
        with open(outdir / f'{prefix}_shapes.json', 'w') as f:
            json.dump(shapes, f, indent=2)
        
        # RMS values for quick comparison
        def rms(t):
            return float(t.float().pow(2).mean().sqrt())
        
        rms_values = {
            'input': rms(self._debug_input),
            'attn_q': rms(self._debug_q),
            'attn_k': rms(self._debug_k),
            'attn_v': rms(self._debug_v),
            'attn_out': rms(self._debug_attn_out),
            'moe_out': rms(self._debug_moe_out),
            'output': rms(self._debug_output),
        }
        with open(outdir / f'{prefix}_rms.json', 'w') as f:
            json.dump(rms_values, f, indent=2)
        
        print(f"Dumped all tensors to {outdir}/")
        print(f"RMS values: {json.dumps(rms_values, indent=2)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=0, help='Layer index to dump')
    parser.add_argument('--shard-dir', type=str, required=True, 
                        help='Directory containing safetensors shards + index')
    parser.add_argument('--seq-len', type=int, default=8, help='Sequence length for test')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--outdir', type=str, default='test_output', help='Output directory')
    args = parser.parse_args()
    
    ref = LayerRef(args.layer, args.shard_dir)
    
    # Create test input
    torch.manual_seed(42)
    hidden_size = ref.config['hidden_size']
    test_input = torch.randn(args.batch, args.seq_len, hidden_size, dtype=torch.float32)
    
    # Run layer
    with torch.no_grad():
        output = ref.forward(test_input)
    
    # Dump
    ref.dump_tensors(args.outdir)


if __name__ == '__main__':
    main()
