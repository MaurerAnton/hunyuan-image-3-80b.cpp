#pragma once

#include <cstdint>

// Patches to add batch support, Conv3d chunking, KV cache to existing components.
// Included from hunyuan_image_gen.h and hunyuan_vae.h to extend functionality.

// ============================================================
// 2D RoPE — batch support
// ============================================================

void build_2d_rope_batch(
    uint32_t batch,
    uint32_t seq_len,
    uint32_t n_elem,
    const uint32_t* text_lens,   // [batch] — text tokens per sample
    const uint32_t* img_hs,      // [batch] — image height per sample
    const uint32_t* img_ws,      // [batch] — image width per sample
    float base,
    float* cos_out,              // [batch, seq_len, n_elem]
    float* sin_out);

// ============================================================
// Conv3d chunking — splits T dimension for memory safety
// ============================================================

void conv3d_chunked_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const struct Conv3dWeights& w,
    float* out,
    float max_memory_gb = 2.0f);

// ============================================================
// Attention with KV cache (integrated into attention_forward_cpu)
// ============================================================

void attention_forward_with_kv_cache_f32(
    const float* hidden_states,   // [1, hidden_size] — single new token
    const struct AttentionWeights& w,
    const struct HunyuanConfig& cfg,
    struct KVCache* cache,
    uint32_t layer_idx,
    float* output);               // [hidden_size]
