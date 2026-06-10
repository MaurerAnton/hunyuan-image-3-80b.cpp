#!/usr/bin/env python3
"""
Convert Hunyuan Image 3.0 safetensors to GGUF format.

Supports:
- BF16 → F32 conversion (default)
- Q8_0 quantization (--type q8_0)
- Per-layer GGUF files for layer-by-layer inference (--per-layer)
- Single combined GGUF (default)

Weights and metadata follow llama.cpp GGUF conventions where applicable,
with custom "hunyuan." prefix for architecture-specific keys.

Usage:
  # Full model to single GGUF (needs all 32 shards + ~300GB RAM for conversion)
  python scripts/convert_to_gguf.py --shard-dir weights/ --output hunyuan-f32.gguf

  # Per-layer GGUF files (each ~5GB, can convert one at a time)
  python scripts/convert_to_gguf.py --shard-dir weights/ --per-layer --output-dir gguf/

  # Single layer with Q8_0 quantization
  python scripts/convert_to_gguf.py --shard-dir weights/ --per-layer --layer 0 --type q8_0 --output-dir gguf/
"""

import argparse, json, os, struct, sys
from pathlib import Path
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType

sys.path.insert(0, str(Path(__file__).parent.parent))
import safetensors.torch


# ============================================================
# Hunyuan-specific GGUF metadata keys
# ============================================================

HUNYUAN_METADATA = {
    # Architecture identifier
    "general.architecture": "hunyuan3",
    "general.name": "HunyuanImage3.0",
    "general.quantization_version": 2,
    
    # Model parameters
    "hunyuan.hidden_size": None,
    "hunyuan.num_hidden_layers": None,
    "hunyuan.num_attention_heads": None,
    "hunyuan.num_key_value_heads": None,
    "hunyuan.attention_head_dim": None,
    "hunyuan.intermediate_size": None,
    "hunyuan.moe_intermediate_size": None,
    "hunyuan.num_experts": None,
    "hunyuan.moe_topk": None,
    "hunyuan.num_shared_expert": None,
    "hunyuan.vocab_size": None,
    "hunyuan.max_position_embeddings": None,
    "hunyuan.rms_norm_eps": None,
    "hunyuan.rope_theta": None,
    "hunyuan.use_qk_norm": None,
    "hunyuan.use_mixed_mlp_moe": None,
    
    # Tokenizer (minimal — for token embedding lookup only)
    "tokenizer.ggml.model": "none",  # No full tokenizer needed
    "tokenizer.ggml.bos_token_id": None,
    "tokenizer.ggml.eos_token_id": None,
}

# Tensor name mapping: safetensors → GGUF
# We keep the same hierarchy but use ggml-friendly names
def tensor_gguf_name(sf_name):
    """Convert safetensors tensor name to GGUF name."""
    # model.wte.weight → token_embd.weight
    if sf_name == "model.wte.weight":
        return "token_embd.weight"
    # model.ln_f.weight → output_norm.weight
    if sf_name == "model.ln_f.weight":
        return "output_norm.weight"
    # lm_head.weight → output.weight
    if sf_name == "lm_head.weight":
        return "output.weight"
    # model.layers.{i}.input_layernorm.weight → blk.{i}.attn_norm.weight
    # model.layers.{i}.post_attention_layernorm.weight → blk.{i}.ffn_norm.weight
    # model.layers.{i}.self_attn.qkv_proj.weight → blk.{i}.attn_qkv.weight
    # model.layers.{i}.self_attn.o_proj.weight → blk.{i}.attn_output.weight
    # model.layers.{i}.self_attn.query_layernorm.weight → blk.{i}.attn_q_norm.weight
    # model.layers.{i}.self_attn.key_layernorm.weight → blk.{i}.attn_k_norm.weight
    # model.layers.{i}.mlp.shared_mlp.gate_and_up_proj.weight → blk.{i}.ffn_shared_gate_up.weight
    # model.layers.{i}.mlp.shared_mlp.down_proj.weight → blk.{i}.ffn_shared_down.weight
    # model.layers.{i}.mlp.gate.wg.weight → blk.{i}.ffn_gate.weight
    # model.layers.{i}.mlp.experts.{e}.gate_and_up_proj.weight → blk.{i}.ffn_expert_{e}_gate_up.weight
    # model.layers.{i}.mlp.experts.{e}.down_proj.weight → blk.{i}.ffn_expert_{e}_down.weight
    
    import re
    
    # Layer patterns
    m = re.match(r'model\.layers\.(\d+)\.(.+)', sf_name)
    if m:
        layer = m.group(1)
        rest = m.group(2)
        
        # Norms
        if rest == 'input_layernorm.weight':
            return f'blk.{layer}.attn_norm.weight'
        if rest == 'post_attention_layernorm.weight':
            return f'blk.{layer}.ffn_norm.weight'
        
        # Attention
        if rest == 'self_attn.qkv_proj.weight':
            return f'blk.{layer}.attn_qkv.weight'
        if rest == 'self_attn.o_proj.weight':
            return f'blk.{layer}.attn_output.weight'
        if rest == 'self_attn.query_layernorm.weight':
            return f'blk.{layer}.attn_q_norm.weight'
        if rest == 'self_attn.key_layernorm.weight':
            return f'blk.{layer}.attn_k_norm.weight'
        
        # MoE shared
        if rest == 'mlp.shared_mlp.gate_and_up_proj.weight':
            return f'blk.{layer}.ffn_shared_gate_up.weight'
        if rest == 'mlp.shared_mlp.down_proj.weight':
            return f'blk.{layer}.ffn_shared_down.weight'
        
        # Router
        if rest == 'mlp.gate.wg.weight':
            return f'blk.{layer}.ffn_gate.weight'
        
        # Experts
        em = re.match(r'mlp\.experts\.(\d+)\.(.+)', rest)
        if em:
            expert = em.group(1)
            etype = em.group(2)
            if etype == 'gate_and_up_proj.weight':
                return f'blk.{layer}.ffn_expert_{expert}_gate_up.weight'
            if etype == 'down_proj.weight':
                return f'blk.{layer}.ffn_expert_{expert}_down.weight'
    
    return sf_name  # fallback


def get_quant_type(type_str):
    """Convert type string to GGMLQuantizationType."""
    mapping = {
        'f32': GGMLQuantizationType.F32,
        'f16': GGMLQuantizationType.F16,
        'q8_0': GGMLQuantizationType.Q8_0,
        'q4_0': GGMLQuantizationType.Q4_0,
        'q4_1': GGMLQuantizationType.Q4_1,
        'q5_0': GGMLQuantizationType.Q5_0,
        'q5_1': GGMLQuantizationType.Q5_1,
    }
    if type_str.lower() not in mapping:
        print(f"Unknown type {type_str}, using F32")
        return GGMLQuantizationType.F32
    return mapping[type_str.lower()]


def convert_layer_to_gguf(shard_dir, index, layer_idx, output_path, quant_type):
    """Convert a single decoder layer's weights to GGUF."""
    cfg = json.load(open(Path(__file__).parent.parent / 'config.json'))
    
    # Collect all tensors for this layer
    prefix = f'model.layers.{layer_idx}.'
    shard_files = set()
    for name in index['weight_map']:
        if name.startswith(prefix):
            shard_files.add(index['weight_map'][name])
    
    # Load tensors from shards
    tensors = {}
    for sf_name in sorted(shard_files):
        sf_path = os.path.join(shard_dir, sf_name)
        if not os.path.exists(sf_path):
            print(f"  SKIP missing: {sf_path}")
            continue
        shard = safetensors.torch.load_file(sf_path)
        for k, v in shard.items():
            if k.startswith(prefix):
                tensors[tensor_gguf_name(k)] = v.float().numpy()
        print(f"  Loaded {len(shard)} tensors from {sf_name}")
    
    # Write GGUF
    writer = GGUFWriter(output_path, "hunyuan3")
    
    # Add metadata
    writer.add_uint32("hunyuan.hidden_size", cfg['hidden_size'])
    writer.add_uint32("hunyuan.num_attention_heads", cfg['num_attention_heads'])
    writer.add_uint32("hunyuan.num_key_value_heads", cfg['num_key_value_heads'])
    writer.add_uint32("hunyuan.attention_head_dim", cfg['attention_head_dim'])
    writer.add_uint32("hunyuan.moe_intermediate_size", cfg['moe_intermediate_size'][layer_idx])
    writer.add_uint32("hunyuan.num_experts", cfg['num_experts'])
    writer.add_uint32("hunyuan.moe_topk", cfg['moe_topk'][layer_idx])
    writer.add_uint32("hunyuan.layer_idx", layer_idx)
    writer.add_float32("hunyuan.rms_norm_eps", cfg['rms_norm_eps'])
    writer.add_float32("hunyuan.rope_theta", cfg['rope_theta'])
    
    # Add tensors (keep PyTorch [out, in] layout for manual matmul loops)
    for name in sorted(tensors.keys()):
        data = tensors[name]
        # NO transpose — manual C++ loops use [out, in] layout
        writer.add_tensor(name, data, raw_dtype=quant_type)
        print(f"  {name}: shape={list(data.shape)}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Wrote {output_path} ({size_mb:.1f} MB)")


def convert_global_to_gguf(shard_dir, index, output_path, quant_type):
    """Convert global weights (embedding, final norm, lm_head) to GGUF."""
    cfg = json.load(open(Path(__file__).parent.parent / 'config.json'))
    
    global_tensors = {
        'model.wte.weight': 'token_embd.weight',
        'model.ln_f.weight': 'output_norm.weight',
        'lm_head.weight': 'output.weight',
    }
    
    writer = GGUFWriter(output_path, "hunyuan3")
    
    # Metadata
    writer.add_uint32("hunyuan.vocab_size", cfg['vocab_size'])
    writer.add_uint32("hunyuan.hidden_size", cfg['hidden_size'])
    writer.add_uint32("hunyuan.num_hidden_layers", cfg['num_hidden_layers'])
    writer.add_float32("hunyuan.rms_norm_eps", cfg['rms_norm_eps'])
    writer.add_uint32("tokenizer.ggml.bos_token_id", cfg.get('bos_token_id', 127958))
    writer.add_uint32("tokenizer.ggml.eos_token_id", cfg.get('eos_token_id', 127957))
    writer.add_string("tokenizer.ggml.model", "none")
    
    writer.add_string("general.architecture", "hunyuan3")
    writer.add_string("general.name", "HunyuanImage3.0")
    
    for sf_name, gguf_name in global_tensors.items():
        shard_name = index['weight_map'][sf_name]
        sf_path = os.path.join(shard_dir, shard_name)
        if not os.path.exists(sf_path):
            print(f"  SKIP: {sf_path}")
            continue
        shard = safetensors.torch.load_file(sf_path)
        data = shard[sf_name].float().numpy()
        # NO transpose - keep PyTorch layout
        writer.add_tensor(gguf_name, data, raw_dtype=quant_type)
        print(f"  {gguf_name}: shape={list(data.shape)}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-dir', required=True, help='Directory with safetensors shards')
    parser.add_argument('--output', '-o', help='Output GGUF file (single-file mode)')
    parser.add_argument('--per-layer', action='store_true', help='Create per-layer GGUF files')
    parser.add_argument('--layer', type=int, default=None, help='Specific layer (with --per-layer)')
    parser.add_argument('--output-dir', default='gguf', help='Output directory for per-layer mode')
    parser.add_argument('--type', default='f32', help='Quantization: f32, f16, q8_0, q4_0')
    args = parser.parse_args()
    
    # Load index
    index_path = os.path.join(args.shard_dir, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(os.path.dirname(args.shard_dir), 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(os.path.dirname(args.shard_dir), 'model_index.json')
    
    with open(index_path) as f:
        index = json.load(f)
    
    quant = get_quant_type(args.type)
    
    if args.per_layer:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert global weights first
        global_path = os.path.join(args.output_dir, 'hunyuan_global.gguf')
        convert_global_to_gguf(args.shard_dir, index, global_path, 
                              GGMLQuantizationType.F32)  # Globals always F32
        
        # Convert layer(s)
        layers = [args.layer] if args.layer is not None else range(32)
        for layer in layers:
            layer_path = os.path.join(args.output_dir, f'hunyuan_layer_{layer}.gguf')
            print(f"\n=== Layer {layer} ===")
            convert_layer_to_gguf(args.shard_dir, index, layer, layer_path, quant)
    
    elif args.output:
        # Single-file mode: convert everything
        convert_global_to_gguf(args.shard_dir, index, args.output, quant)
        for layer in range(32):
            print(f"\n=== Layer {layer} ===")
            # Append layer tensors to the same file
            # TODO: implement append mode for GGUF
            print("  Single-file mode with all layers not yet implemented.")
            print("  Use --per-layer for now.")
            break
    else:
        print("Specify --output or --per-layer")


if __name__ == '__main__':
    main()
