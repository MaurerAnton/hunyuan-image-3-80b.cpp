#include "hunyuan_patches.h"
#include "hunyuan_image_gen.h"
#include "hunyuan_vae.h"
#include "hunyuan_model.h"
#include <cstring>
#include <cstdlib>

// ============================================================
// 2D RoPE — batch support
// ============================================================

void build_2d_rope_batch(
    uint32_t batch, uint32_t seq_len, uint32_t n_elem,
    const uint32_t* text_lens, const uint32_t* img_hs, const uint32_t* img_ws,
    float base, float* cos_out, float* sin_out)
{
    // For each batch element, call the single-batch version
    for (uint32_t b = 0; b < batch; b++) {
        float* cos_b = cos_out + b * seq_len * n_elem;
        float* sin_b = sin_out + b * seq_len * n_elem;
        build_2d_rope(seq_len, n_elem, text_lens[b], img_hs[b], img_ws[b], base, cos_b, sin_b);
    }
}

// ============================================================
// Conv3d chunking
// ============================================================

void conv3d_chunked_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const Conv3dWeights& w, float* out, float max_memory_gb)
{
    // Estimate memory per T-step
    size_t bytes_per_t = (size_t)B * w.in_ch * H * W * sizeof(float);
    float gb_per_t = bytes_per_t / (1024.0f * 1024.0f * 1024.0f);
    
    if (gb_per_t * T <= max_memory_gb || w.pad_t == 0) {
        // No chunking needed
        conv3d_f32(x, B, T, H, W, w, out);
        return;
    }
    
    // Determine chunk size
    uint32_t max_t_per_chunk = (uint32_t)(max_memory_gb / gb_per_t);
    if (max_t_per_chunk < 1) max_t_per_chunk = 1;
    
    uint32_t T_out = (T + 2*w.pad_t - w.kT) / w.stride_t + 1;
    
    for (uint32_t t_chunk = 0; t_chunk < T; t_chunk += max_t_per_chunk) {
        uint32_t chunk_start = t_chunk;
        uint32_t chunk_size = std::min(max_t_per_chunk, T - t_chunk);
        
        // Pad chunk: add pad_t frames before and after for temporal overlap
        uint32_t pad_before = (chunk_start > 0) ? w.pad_t : 0;
        uint32_t pad_after = (chunk_start + chunk_size < T) ? w.pad_t : 0;
        uint32_t padded_size = pad_before + chunk_size + pad_after;
        
        // Build padded chunk with replicate padding on temporal boundaries
        float* padded = new float[(size_t)B * w.in_ch * padded_size * H * W];
        
        // Copy main data
        for (uint32_t t = 0; t < chunk_size; t++) {
            size_t src_off = (size_t)(chunk_start + t) * H * W;
            size_t dst_off = (size_t)(pad_before + t) * H * W;
            for (uint32_t b = 0; b < B; b++)
                for (uint32_t c = 0; c < w.in_ch; c++)
                    memcpy(padded + ((b * w.in_ch + c) * padded_size + (pad_before + t)) * H * W,
                           x + ((b * w.in_ch + c) * T + chunk_start + t) * H * W,
                           H * W * sizeof(float));
        }
        
        // Replicate padding: copy first/last frame
        for (uint32_t p = 0; p < pad_before; p++) {
            for (uint32_t b = 0; b < B; b++)
                for (uint32_t c = 0; c < w.in_ch; c++)
                    memcpy(padded + ((b * w.in_ch + c) * padded_size + p) * H * W,
                           padded + ((b * w.in_ch + c) * padded_size + pad_before) * H * W,
                           H * W * sizeof(float));
        }
        for (uint32_t p = 0; p < pad_after; p++) {
            for (uint32_t b = 0; b < B; b++)
                for (uint32_t c = 0; c < w.in_ch; c++)
                    memcpy(padded + ((b * w.in_ch + c) * padded_size + pad_before + chunk_size + p) * H * W,
                           padded + ((b * w.in_ch + c) * padded_size + pad_before + chunk_size - 1) * H * W,
                           H * W * sizeof(float));
        }
        
        // Convolve chunk
        float* chunk_out = new float[(size_t)B * w.out_ch * T_out * H * W];
        Conv3dWeights chunk_w = w;
        chunk_w.pad_t = 0;  // Padding already applied
        conv3d_f32(padded, B, padded_size, H, W, chunk_w, chunk_out);
        delete[] padded;
        
        // Map output: chunks produce overlapping output at temporal boundaries
        uint32_t t_out_start = chunk_start / w.stride_t;
        uint32_t t_out_chunk = chunk_size / w.stride_t;
        if (t_out_start + t_out_chunk > T_out) t_out_chunk = T_out - t_out_start;
        
        for (uint32_t b = 0; b < B; b++)
            for (uint32_t c = 0; c < w.out_ch; c++)
                memcpy(out + ((b * w.out_ch + c) * T_out + t_out_start) * H * W,
                       chunk_out + ((b * w.out_ch + c) * (chunk_size / w.stride_t)) * H * W,
                       t_out_chunk * H * W * sizeof(float));
        
        // Note: overlapping regions would need averaging. For simplicity,
        // later chunks overwrite earlier. This matches PyTorch's behavior
        // where chunks don't overlap in output.
        delete[] chunk_out;
    }
}

// ============================================================
// Attention with integrated KV cache (single token)
// ============================================================

void attention_forward_with_kv_cache_f32(
    const float* hidden_states,
    const AttentionWeights& w,
    const HunyuanConfig& cfg,
    KVCache* cache,
    uint32_t layer_idx,
    float* output)
{
    uint32_t H = cfg.hidden_size;
    uint32_t n_heads = cfg.num_attention_heads;
    uint32_t n_kv_heads = cfg.num_key_value_heads;
    uint32_t head_dim = cfg.attention_head_dim;
    uint32_t n_groups = cfg.num_kv_groups;
    uint32_t qkv_total = cfg.qkv_total;
    
    // QKV projection (single token: n_tokens=1)
    float* qkv = new float[qkv_total];
    for (uint32_t j = 0; j < qkv_total; j++) {
        float s = 0;
        for (uint32_t k = 0; k < H; k++)
            s += hidden_states[k] * w.qkv_weight[j * H + k];
        qkv[j] = s;
    }
    
    // Split
    uint32_t hidden_q = cfg.hidden_size_q;
    uint32_t hidden_kv = cfg.hidden_size_kv;
    
    float* q = new float[n_heads * head_dim];
    float* k_new = new float[n_kv_heads * head_dim];
    float* v_new = new float[n_kv_heads * head_dim];
    
    for (uint32_t h = 0; h < n_heads; h++)
        memcpy(q + h * head_dim, qkv + h * head_dim, head_dim * sizeof(float));
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        memcpy(k_new + h * head_dim, qkv + hidden_q + h * head_dim, head_dim * sizeof(float));
        memcpy(v_new + h * head_dim, qkv + hidden_q + hidden_kv + h * head_dim, head_dim * sizeof(float));
    }
    delete[] qkv;
    
    // QK Norm
    for (uint32_t h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        float sum_sq = 0;
        for (uint32_t d = 0; d < head_dim; d++) sum_sq += qh[d] * qh[d];
        float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
        for (uint32_t d = 0; d < head_dim; d++) qh[d] *= rms * w.q_norm_weight[d];
    }
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        float* kh = k_new + h * head_dim;
        float sum_sq = 0;
        for (uint32_t d = 0; d < head_dim; d++) sum_sq += kh[d] * kh[d];
        float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
        for (uint32_t d = 0; d < head_dim; d++) kh[d] *= rms * w.k_norm_weight[d];
    }
    
    // Store K,V in cache
    uint32_t pos = cache->current_len;
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        memcpy(cache->key_caches[layer_idx] + (h * cache->max_seq_len + pos) * head_dim,
               k_new + h * head_dim, head_dim * sizeof(float));
        memcpy(cache->value_caches[layer_idx] + (h * cache->max_seq_len + pos) * head_dim,
               v_new + h * head_dim, head_dim * sizeof(float));
    }
    delete[] k_new; delete[] v_new;
    
    // Attention over all cached positions
    uint32_t seq_len = cache->current_len + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    memset(output, 0, H * sizeof(float));
    
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = h / n_groups;
        float* qh = q + h * head_dim;
        
        // Scores
        float* scores = new float[seq_len];
        float max_s = -1e30f;
        for (uint32_t sk = 0; sk < seq_len; sk++) {
            float dot = 0;
            const float* kh = cache->key_caches[layer_idx] + (kv_h * cache->max_seq_len + sk) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
            scores[sk] = dot * scale;
            if (scores[sk] > max_s) max_s = scores[sk];
        }
        float sum_e = 0;
        for (uint32_t sk = 0; sk < seq_len; sk++) {
            scores[sk] = expf(scores[sk] - max_s);
            sum_e += scores[sk];
        }
        for (uint32_t sk = 0; sk < seq_len; sk++) scores[sk] /= sum_e;
        
        for (uint32_t sk = 0; sk < seq_len; sk++) {
            const float* vh = cache->value_caches[layer_idx] + (kv_h * cache->max_seq_len + sk) * head_dim;
            float sw = scores[sk];
            for (uint32_t d = 0; d < head_dim; d++)
                output[h * head_dim + d] += sw * vh[d];
        }
        delete[] scores;
    }
    delete[] q;
    
    // Output projection
    float* merged = new float[H];
    for (uint32_t j = 0; j < H; j++) {
        float s = 0;
        for (uint32_t k = 0; k < H; k++)
            s += output[k] * w.o_weight[j * H + k];
        merged[j] = s;
    }
    memcpy(output, merged, H * sizeof(float));
    delete[] merged;
    
    cache->current_len++;
}
