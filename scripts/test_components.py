#!/usr/bin/env python3
"""
Component verification: 2D RoPE, TimestepEmbedder, UNet blocks.
Tests each component individually against PyTorch reference.

Usage:
  python scripts/test_components.py --component rope_2d --shard-dir weights/
  python scripts/test_components.py --component timestep --shard-dir weights/
  python scripts/test_components.py --component unet_down --shard-dir weights/
  python scripts/test_components.py --component unet_up --shard-dir weights/
"""

import argparse, json, os, struct, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import safetensors.torch


def load_config():
    with open(Path(__file__).parent.parent / 'config.json') as f:
        return json.load(f)


def load_tensor(shard_dir, index, tensor_name):
    """Load a single tensor from safetensors shards."""
    with open(index) as f:
        idx = json.load(f)
    shard_name = idx['weight_map'][tensor_name]
    shard_path = os.path.join(shard_dir, shard_name)
    shard = safetensors.torch.load_file(shard_path)
    return shard[tensor_name].float()


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def correlation(a, b):
    a = a.flatten().numpy() if isinstance(a, torch.Tensor) else a.flatten()
    b = b.flatten().numpy() if isinstance(b, torch.Tensor) else b.flatten()
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# 2D RoPE Test
# ============================================================

def test_rope_2d(shard_dir, index_path, outdir):
    cfg = load_config()
    head_dim = cfg['attention_head_dim']
    n_heads = cfg['num_attention_heads']
    n_kv_heads = cfg['num_key_value_heads']
    base = cfg['rope_theta']
    
    # Test: 4 text tokens + 2×2 image patches = 8 tokens
    text_len = 4
    img_h, img_w = 2, 2
    seq_len = text_len + img_h * img_w
    
    # Create random Q, K
    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_kv_heads, seq_len, head_dim)
    
    # Python 2D RoPE
    from hunyuan import build_2d_rope, apply_rotary_pos_emb
    
    # Create image_infos for 2D RoPE
    image_infos = [[(slice(text_len, seq_len), (img_h, img_w))]]
    cos, sin = build_2d_rope(seq_len, head_dim, image_infos=image_infos, 
                              device=q.device, base=base)
    cos = cos.unsqueeze(0)  # add batch dim
    sin = sin.unsqueeze(0)
    
    q_rot_py, k_rot_py = apply_rotary_pos_emb(q, k, cos, sin)
    
    # Save for C++ comparison
    q.numpy().astype(np.float32).tofile(f'{outdir}/rope_q_input.bin')
    k.numpy().astype(np.float32).tofile(f'{outdir}/rope_k_input.bin')
    q_rot_py.numpy().astype(np.float32).tofile(f'{outdir}/rope_q_py.bin')
    k_rot_py.numpy().astype(np.float32).tofile(f'{outdir}/rope_k_py.bin')
    
    # Also save cos/sin for C++ to use
    cos[0].numpy().astype(np.float32).tofile(f'{outdir}/rope_cos_py.bin')
    sin[0].numpy().astype(np.float32).tofile(f'{outdir}/rope_sin_py.bin')
    
    # Save test params
    with open(f'{outdir}/rope_params.json', 'w') as f:
        json.dump({'text_len': text_len, 'img_h': img_h, 'img_w': img_w,
                   'seq_len': seq_len, 'head_dim': head_dim, 'n_heads': n_heads,
                   'n_kv_heads': n_kv_heads, 'base': base}, f)
    
    print(f"2D RoPE: Q RMS={rms(q_rot_py):.6f} K RMS={rms(k_rot_py):.6f}")
    print(f"  Q input RMS={rms(q):.6f} → output RMS={rms(q_rot_py):.6f}")
    print(f"Dumped to {outdir}/")


# ============================================================
# Timestep Embedder Test
# ============================================================

def test_timestep(shard_dir, index_path, outdir):
    cfg = load_config()
    
    # Load timestep_emb weights
    # TimestepEmbedder fields: mlp[0] (Linear), mlp[1] (GELU), mlp[2] (Linear)
    # Weights stored in model.safetensors.index.json
    prefix = 'timestep_emb.mlp.'
    indices = ['0.weight', '0.bias', '2.weight', '2.bias']
    
    w0 = load_tensor(shard_dir, index_path, prefix + '0.weight')
    b0 = load_tensor(shard_dir, index_path, prefix + '0.bias')
    w2 = load_tensor(shard_dir, index_path, prefix + '2.weight')
    b2 = load_tensor(shard_dir, index_path, prefix + '2.bias')
    
    # Save weights for C++
    w0.numpy().tofile(f'{outdir}/timestep_w0.bin')
    b0.numpy().tofile(f'{outdir}/timestep_b0.bin')
    w2.numpy().tofile(f'{outdir}/timestep_w2.bin')
    b2.numpy().tofile(f'{outdir}/timestep_b2.bin')
    
    # Test with random timesteps
    torch.manual_seed(42)
    t = torch.rand(4) * 1000  # 4 timesteps in [0, 1000]
    t.numpy().tofile(f'{outdir}/timestep_t.bin')
    
    # Python forward
    from hunyuan import timestep_embedding
    freq_dim = 256
    max_period = 10000
    t_freq = timestep_embedding(t, freq_dim, max_period).type(w0.dtype)
    hidden = F.linear(t_freq, w0, b0)
    hidden = F.gelu(hidden)
    out = F.linear(hidden, w2, b2)
    
    out.numpy().tofile(f'{outdir}/timestep_py.bin')
    
    with open(f'{outdir}/timestep_params.json', 'w') as f:
        json.dump({'batch': 4, 'freq_dim': freq_dim, 'max_period': max_period,
                   'hidden_size': w0.shape[0], 'out_size': w2.shape[0]}, f)
    
    print(f"TimestepEmbedder: output RMS={rms(out):.6f}")
    print(f"  Weights: w0={list(w0.shape)} w2={list(w2.shape)}")
    print(f"Dumped to {outdir}/")


# ============================================================
# UNetDown (patch_embed) Test
# ============================================================

def test_unet_down(shard_dir, index_path, outdir):
    cfg = load_config()
    
    # Load all patch_embed weights
    sf_tensors = {}
    with open(index_path) as f:
        idx = json.load(f)
    for name in idx['weight_map']:
        if name.startswith('patch_embed.'):
            sf_tensors[name] = load_tensor(shard_dir, index_path, name)
    
    # Also load time_embed for conditioning
    for name in idx['weight_map']:
        if name.startswith('time_embed.mlp.'):
            sf_tensors[name] = load_tensor(shard_dir, index_path, name)
    
    H, W = 8, 8  # Small test latent
    torch.manual_seed(42)
    x = torch.randn(1, 32, H, W)  # VAE latent
    t = torch.tensor([500.0])       # single timestep
    
    x.numpy().tofile(f'{outdir}/unet_down_x.bin')
    t.numpy().tofile(f'{outdir}/unet_down_t.bin')
    
    # Python forward using actual model code
    from hunyuan import TimestepEmbedder, UNetDown
    
    # Build TimestepEmbedder with loaded weights
    te = TimestepEmbedder(hidden_size=cfg['hidden_size'])
    te.mlp[0].weight.data = sf_tensors['time_embed.mlp.0.weight']
    te.mlp[0].bias.data = sf_tensors['time_embed.mlp.0.bias']
    te.mlp[2].weight.data = sf_tensors['time_embed.mlp.2.weight']
    te.mlp[2].bias.data = sf_tensors['time_embed.mlp.2.bias']
    
    t_emb = te(t)
    t_emb.numpy().tofile(f'{outdir}/unet_down_temb.bin')
    
    # Build UNetDown
    ud = UNetDown(patch_size=cfg.get('patch_size', 1),
                  emb_channels=cfg['hidden_size'],
                  in_channels=cfg['vae']['latent_channels'],
                  hidden_channels=cfg.get('patch_embed_hidden_dim', 1024),
                  out_channels=cfg['hidden_size'])
    
    # Load weights into UNetDown
    for pname, param in ud.named_parameters():
        key = f'patch_embed.{pname}'
        if key in sf_tensors:
            param.data = sf_tensors[key]
    
    # Forward
    with torch.no_grad():
        out, _, _ = ud(x, t_emb)
    
    out.numpy().tofile(f'{outdir}/unet_down_py.bin')
    
    print(f"UNetDown: input [1,32,{H},{W}] RMS={rms(x):.6f}")
    print(f"  output {list(out.shape)} RMS={rms(out):.6f}")
    print(f"  t_emb RMS={rms(t_emb):.6f}")
    
    # Save weight info for C++
    weight_info = {}
    for pname, param in ud.named_parameters():
        key = f'patch_embed.{pname}'
        weight_info[key] = list(param.shape)
        out_name = f'unet_down_{pname.replace(".", "_")}'
        param.data.numpy().tofile(f'{outdir}/{out_name}.bin')
    
    with open(f'{outdir}/unet_down_params.json', 'w') as f:
        json.dump({'H': H, 'W': W, 'in_ch': 32, 'out_ch': cfg['hidden_size'],
                   'hid_ch': cfg.get('patch_embed_hidden_dim', 1024),
                   'emb_ch': cfg['hidden_size'], 'weights': weight_info}, f)
    
    print(f"  Dumped {len(weight_info)} weight tensors")
    print(f"Dumped to {outdir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', required=True, 
                        choices=['rope_2d', 'timestep', 'unet_down', 'unet_up'])
    parser.add_argument('--shard-dir', required=True)
    parser.add_argument('--outdir', default='test_components')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Find index
    index_path = os.path.join(args.shard_dir, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(os.path.dirname(args.shard_dir), 'model.safetensors.index.json')
    
    if args.component == 'rope_2d':
        test_rope_2d(args.shard_dir, index_path, args.outdir)
    elif args.component == 'timestep':
        test_timestep(args.shard_dir, index_path, args.outdir)
    elif args.component == 'unet_down':
        test_unet_down(args.shard_dir, index_path, args.outdir)
    elif args.component == 'unet_up':
        print("UNetUp test — TODO")


if __name__ == '__main__':
    main()
