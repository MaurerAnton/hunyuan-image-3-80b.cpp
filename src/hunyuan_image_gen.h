#pragma once

#include "hunyuan_model.h"
#include <cstdint>
#include <cmath>
#include <cstring>

// ============================================================
// 2D Rotary Position Embedding
// 
// Hunyuan uses 2D RoPE where image patches get grid positions
// and text tokens get sequential 1D positions.
// ============================================================

void build_2d_rope(
    uint32_t seq_len,
    uint32_t n_elem,           // head_dim (128)
    uint32_t text_len,         // number of text tokens before image
    uint32_t img_h,            // image patches height
    uint32_t img_w,            // image patches width
    float base,                // 10000.0
    float* cos_out,            // [seq_len, n_elem]
    float* sin_out);           // [seq_len, n_elem]

// Apply RoPE to Q and K tensors using precomputed cos/sin
void apply_rope_2d_f32(
    float* q,                  // [n_tokens, n_heads, head_dim]
    float* k,                  // [n_tokens, n_kv_heads, head_dim]
    const float* cos,          // [n_tokens, head_dim]
    const float* sin,          // [n_tokens, head_dim]
    uint32_t n_tokens,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim);

// ============================================================
// Timestep Embedder
//
// Sinusoidal embedding → Linear → GELU → Linear
// Input: scalar timestep t, Output: vector [hidden_size]
// ============================================================

struct TimestepEmbedderWeights {
    std::vector<float> mlp_0_weight;  // [hidden_size, 256] — Linear 256→hidden
    std::vector<float> mlp_0_bias;    // [hidden_size]
    std::vector<float> mlp_2_weight;  // [out_size, hidden_size] — Linear hidden→out
    std::vector<float> mlp_2_bias;    // [out_size]
    
    uint32_t frequency_embedding_size = 256;
    uint32_t hidden_size = 4096;
    uint32_t out_size = 4096;
    float max_period = 10000.0f;
};

void timestep_embedding_f32(
    const float* t,                   // [batch] scalar timesteps
    uint32_t batch,
    const TimestepEmbedderWeights& w,
    float* out);                      // [batch, out_size]

// ============================================================
// ResBlock (from UNet)
//
// GroupNorm → SiLU → Conv3×3 (+ timestep conditioning via scale/shift)
// ============================================================

struct ResBlockWeights {
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t emb_channels;    // timestep embedding channels
    
    // in_layers: norm + conv
    // norm: GroupNorm(32, in_channels) → weight[in_channels], bias[in_channels]
    std::vector<float> norm_weight;
    std::vector<float> norm_bias;
    // conv: Conv2d(in_channels, out_channels, 3, padding=1)
    std::vector<float> conv_weight;  // [out_channels, in_channels, 3, 3]
    std::vector<float> conv_bias;    // [out_channels]
    
    // emb_layers: Linear(emb_channels → 2*out_channels) for scale+shift
    std::vector<float> emb_weight;   // [2*out_channels, emb_channels]
    std::vector<float> emb_bias;     // [2*out_channels]
    
    // skip connection (if in_channels != out_channels)
    bool use_conv_shortcut = false;
    std::vector<float> shortcut_weight; // [out_channels, in_channels, 1, 1] or [out_channels, in_channels, 3, 3]
    std::vector<float> shortcut_bias;   // [out_channels]
};

void resblock_forward_f32(
    const float* x,              // [batch, in_ch, H, W]
    const float* emb,            // [batch, emb_ch] timestep embedding
    uint32_t batch,
    uint32_t H, uint32_t W,
    const ResBlockWeights& w,
    float* out);                 // [batch, out_ch, H, W]

// ============================================================
// Downsample / Upsample
// ============================================================

void downsample_conv_f32(
    const float* x,              // [B, C, H, W]
    uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    const float* conv_weight,    // [C, C, 3, 3]
    const float* conv_bias,      // [C]
    float* out);                 // [B, C, H/2, W/2]

void upsample_nearest_f32(
    const float* x,              // [B, C, H, W]
    uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    float* out);                 // [B, C, H*2, W*2]

// ============================================================
// UNetDown (patch_embed)
//
// Converts VAE latents [32, H, W] → transformer tokens [4096, H, W]
// with timestep conditioning.
// ============================================================

struct UNetDownWeights {
    uint32_t in_channels = 32;      // VAE latent channels
    uint32_t out_channels = 4096;   // transformer hidden size
    uint32_t hidden_channels = 1024;
    uint32_t emb_channels = 4096;
    
    // time_emb_proj: Linear(emb_channels → hidden_channels)
    std::vector<float> time_emb_proj_weight;
    std::vector<float> time_emb_proj_bias;
    
    // input_blocks: list of ResBlocks + optional Downsample
    // Block layout: ResBlock(in_ch+emb→hidden), ResBlock(hidden→hidden), Downsample(hidden→hidden)
    //               ResBlock(hidden+emb→hidden), ResBlock(hidden→hidden), Downsample(hidden→hidden)
    //               ResBlock(hidden+emb→hidden), ResBlock(hidden→out)
    // For patch_size=1: no downsample, 6 ResBlocks total
    std::vector<ResBlockWeights> blocks;
    bool has_downsample[3] = {false, false, false}; // patch_size=1 → no downsample
};

void unet_down_forward_f32(
    const float* x,              // [B, in_ch, H, W] — VAE latent
    const float* timestep_emb,   // [B, emb_ch] — time embedding
    uint32_t B, uint32_t H, uint32_t W,
    const UNetDownWeights& w,
    float* out);                 // [B, out_ch, H, W]

// ============================================================
// UNetUp (final_layer)
//
// Converts transformer hidden states [4096, H, W] → VAE latents [32, H, W]
// ============================================================

struct UNetUpWeights {
    uint32_t in_channels = 4096;
    uint32_t out_channels = 32;
    uint32_t hidden_channels = 1024;
    uint32_t emb_channels = 4096;
    
    std::vector<float> time_emb_proj_weight;
    std::vector<float> time_emb_proj_bias;
    
    std::vector<ResBlockWeights> blocks;
    bool out_norm = true;
    std::vector<float> out_norm_weight;  // GroupNorm weight
    std::vector<float> out_norm_bias;    // GroupNorm bias
};

void unet_up_forward_f32(
    const float* x,              // [B, in_ch, H, W]
    const float* timestep_emb,   // [B, emb_ch]
    uint32_t B, uint32_t H, uint32_t W,
    const UNetUpWeights& w,
    float* out);                 // [B, out_ch, H, W]

// ============================================================
// KV Cache for autoregressive generation
// ============================================================

struct KVCache {
    // Per layer: key_cache, value_cache
    // Shape: [n_layers][n_kv_heads][max_seq_len][head_dim]
    std::vector<float*> key_caches;
    std::vector<float*> value_caches;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;
    uint32_t current_len = 0;
    
    void init(uint32_t layers, uint32_t kv_heads, uint32_t hdim, uint32_t max_len);
    void reset();
    ~KVCache();
};

void attention_with_cache_f32(
    const float* q,              // [1, n_heads, head_dim] — single new token
    KVCache& cache,
    uint32_t layer_idx,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float scale,
    const float* o_weight,       // [hidden, n_heads*head_dim]
    float* output);              // [hidden]

// ============================================================
// GroupNorm (used in ResBlock)
// ============================================================

void group_norm_f32(
    const float* x,              // [B, C, H, W]
    uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    uint32_t num_groups,         // 32
    const float* weight,         // [C]
    const float* bias,           // [C]
    float eps,
    float* out);                 // [B, C, H, W]

// ============================================================
// Conv2d (used throughout UNet and VAE)
// ============================================================

void conv2d_f32(
    const float* x,              // [B, in_ch, H, W]
    uint32_t B, uint32_t in_ch, uint32_t H, uint32_t W,
    const float* weight,         // [out_ch, in_ch, kH, kW]
    const float* bias,           // [out_ch]
    uint32_t out_ch,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding,
    float* out);                 // [B, out_ch, H_out, W_out]
