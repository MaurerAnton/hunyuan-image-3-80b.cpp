#!/usr/bin/env python3
"""Verify image gen components with real weights. Self-contained — no hunyuan.py imports."""

import os, sys, json, struct, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import safetensors.torch

# ============================================================
# Copy of TimestepEmbedder from hunyuan.py (lines 546-577)
# ============================================================

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, out_size=None):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None: out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)

# ============================================================
# Copy of UNet blocks from hunyuan.py (lines 580-873)
# ============================================================

def conv_nd(dims, *args, **kwargs):
    if dims == 1: return nn.Conv1d(*args, **kwargs)
    elif dims == 2: return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def normalization(channels):
    return nn.GroupNorm(32, channels)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv: x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)
    def forward(self, x):
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels=None, dropout=0.0,
                 use_conv=False, dims=2, up=False, down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.in_layers = nn.Sequential(
            normalization(self.in_channels), nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(self.in_channels, False, dims)
            self.x_upd = Upsample(self.in_channels, False, dims)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims)
            self.x_upd = Downsample(self.in_channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels, bias=True))
        self.out_layers = nn.Sequential(
            normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, in_channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, in_channels, self.out_channels, 1)
    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_layers(h * (1 + scale) + shift)
        return self.skip_connection(x) + h

class UNetDown(nn.Module):
    def __init__(self, patch_size, emb_channels, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.input_blocks = nn.ModuleList([
            nn.ModuleList([ResBlock(in_channels + emb_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, out_channels)]),
        ])
        self.time_emb_proj = nn.Sequential(nn.Linear(emb_channels, hidden_channels), nn.SiLU())
        if patch_size > 1:
            self.down_levels = [True, True, True, False, False, False]
            self.downsample = nn.ModuleList([
                Downsample(hidden_channels, True, dims=2) for _ in range(3)
            ])
    def forward(self, x, t_emb):
        h, w = x.shape[-2:]
        emb = self.time_emb_proj(t_emb)
        skips = []
        for i, block in enumerate(self.input_blocks):
            if i == 0:
                emb_expanded = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                block_input = torch.cat([x, emb_expanded], dim=1)
                x = block[0](block_input, t_emb)
            else:
                x = block[0](x, t_emb)
            if self.patch_size > 1 and i < 3:
                skips.append(x)
                x = self.downsample[i](x)
        return x, skips, (h, w)

class UNetUp(nn.Module):
    def __init__(self, patch_size, emb_channels, in_channels, hidden_channels, out_channels, out_norm=False):
        super().__init__()
        self.patch_size = patch_size
        self.out_norm = out_norm
        self.output_blocks = nn.ModuleList([
            nn.ModuleList([ResBlock(in_channels + emb_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, hidden_channels)]),
            nn.ModuleList([ResBlock(hidden_channels, hidden_channels, out_channels)]),
        ])
        self.time_emb_proj = nn.Sequential(nn.Linear(emb_channels, hidden_channels), nn.SiLU())
        if patch_size > 1:
            self.up_levels = [False, False, False, True, True, True]
            self.upsample = nn.ModuleList([Upsample(hidden_channels, True, dims=2) for _ in range(3)])
        if out_norm:
            self.final_norm = nn.GroupNorm(32, out_channels)
    def forward(self, x, t_emb):
        emb = self.time_emb_proj(t_emb)
        for i, block in enumerate(self.output_blocks):
            if i == 0:
                emb_expanded = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
                x = torch.cat([x, emb_expanded], dim=1)
            x = block[0](x, t_emb)
            if self.patch_size > 1 and self.up_levels[i]:
                x = self.upsample[i](x)
        if self.out_norm:
            x = self.final_norm(x)
            x = F.silu(x)
        return x

# ============================================================
# Helpers
# ============================================================

def rms(t):
    return float(t.float().pow(2).mean().sqrt())

def load_tensors_with_prefix(shard_dir, index, prefix):
    with open(index) as f:
        idx = json.load(f)
    tensors = {}
    needed_shards = set()
    for k, s in idx['weight_map'].items():
        if k.startswith(prefix):
            needed_shards.add(s)
    for sname in sorted(needed_shards):
        spath = os.path.join(shard_dir, sname)
        if not os.path.exists(spath): continue
        shard = safetensors.torch.load_file(spath)
        for k, v in shard.items():
            if k.startswith(prefix):
                tensors[k] = v.float()
    return tensors

def load_weights_into(module, prefix, all_tensors):
    state = {}
    for pname, param in module.named_parameters():
        key = f'{prefix}.{pname}'
        if key in all_tensors:
            state[pname] = all_tensors[key]
    module.load_state_dict(state, strict=False)

# ============================================================
# Tests
# ============================================================

def test_timestep(all_t, outdir):
    cfg = json.load(open(Path(__file__).parent.parent / 'config.json'))
    te = TimestepEmbedder(hidden_size=cfg['hidden_size'])
    load_weights_into(te, 'timestep_emb', all_t)
    te.eval()
    
    for key in ['timestep_emb.mlp.0.weight', 'timestep_emb.mlp.0.bias',
                'timestep_emb.mlp.2.weight', 'timestep_emb.mlp.2.bias']:
        all_t[key].numpy().tofile(f'{outdir}/timestep_{key.replace(".","_")}.bin')
    
    torch.manual_seed(42)
    t = torch.tensor([0.0, 250.0, 500.0, 999.0])
    t.numpy().tofile(f'{outdir}/timestep_input.bin')
    with torch.no_grad():
        out = te(t)
    out.numpy().tofile(f'{outdir}/timestep_py.bin')
    print(f"TimestepEmbedder: RMS={rms(out):.6f} shape={list(out.shape)}")

def test_unet_down(all_t, outdir):
    cfg = json.load(open(Path(__file__).parent.parent / 'config.json'))
    ps = cfg.get('patch_size', 1)
    ud = UNetDown(ps, cfg['hidden_size'], cfg['vae']['latent_channels'],
                  cfg.get('patch_embed_hidden_dim', 1024), cfg['hidden_size'])
    load_weights_into(ud, 'patch_embed', all_t)
    ud.eval()
    
    te = TimestepEmbedder(hidden_size=cfg['hidden_size'])
    load_weights_into(te, 'time_embed', all_t)
    te.eval()
    
    # Save weights
    for pname, p in ud.named_parameters():
        p.data.numpy().tofile(f'{outdir}/unet_down_{pname.replace(".","_")}.bin')
    for pname, p in te.named_parameters():
        p.data.numpy().tofile(f'{outdir}/unet_down_te_{pname.replace(".","_")}.bin')
    
    H, W = 4, 4
    torch.manual_seed(42)
    x = torch.randn(1, 32, H, W)
    t = torch.tensor([500.0])
    x.numpy().tofile(f'{outdir}/unet_down_input.bin')
    t.numpy().tofile(f'{outdir}/unet_down_t.bin')
    with torch.no_grad():
        t_emb = te(t)
        out, _, _ = ud(x, t_emb)
    t_emb.numpy().tofile(f'{outdir}/unet_down_temb_py.bin')
    out.numpy().tofile(f'{outdir}/unet_down_output_py.bin')
    print(f"UNetDown: RMS_in={rms(x):.6f} RMS_out={rms(out):.6f} shape={list(out.shape)}")

def test_unet_up(all_t, outdir):
    cfg = json.load(open(Path(__file__).parent.parent / 'config.json'))
    ps = cfg.get('patch_size', 1)
    uu = UNetUp(ps, cfg['hidden_size'], cfg['hidden_size'],
                cfg.get('patch_embed_hidden_dim', 1024), cfg['vae']['latent_channels'], True)
    load_weights_into(uu, 'final_layer', all_t)
    uu.eval()
    
    te = TimestepEmbedder(hidden_size=cfg['hidden_size'])
    load_weights_into(te, 'time_embed_2', all_t)
    te.eval()
    
    for pname, p in uu.named_parameters():
        p.data.numpy().tofile(f'{outdir}/unet_up_{pname.replace(".","_")}.bin')
    for pname, p in te.named_parameters():
        p.data.numpy().tofile(f'{outdir}/unet_up_te_{pname.replace(".","_")}.bin')
    
    H, W = 4, 4
    torch.manual_seed(42)
    x = torch.randn(1, cfg['hidden_size'], H, W)
    t = torch.tensor([500.0])
    x.numpy().tofile(f'{outdir}/unet_up_input.bin')
    t.numpy().tofile(f'{outdir}/unet_up_t.bin')
    with torch.no_grad():
        t_emb = te(t)
        out = uu(x, t_emb)
    t_emb.numpy().tofile(f'{outdir}/unet_up_temb_py.bin')
    out.numpy().tofile(f'{outdir}/unet_up_output_py.bin')
    print(f"UNetUp: RMS_in={rms(x):.6f} RMS_out={rms(out):.6f} shape={list(out.shape)}")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-dir', required=True)
    parser.add_argument('--outdir', default='test_image_gen')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    index_path = os.path.join(args.shard_dir, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(os.path.dirname(args.shard_dir), 'model.safetensors.index.json')
    
    all_t = {}
    for pfx in ['timestep_emb.mlp', 'time_embed.mlp', 'time_embed_2.mlp', 'patch_embed.', 'final_layer.']:
        tensors = load_tensors_with_prefix(args.shard_dir, index_path, pfx)
        all_t.update(tensors)
        print(f"  {pfx}: {len(tensors)} tensors")
    print(f"Total: {len(all_t)} tensors\n")
    
    test_timestep(all_t, args.outdir)
    test_unet_down(all_t, args.outdir)
    test_unet_up(all_t, args.outdir)
    print(f"\nDone. Output in {args.outdir}/")

if __name__ == '__main__':
    main()
