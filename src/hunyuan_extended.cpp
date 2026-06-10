#include "hunyuan_extended.h"
#include <cstdlib>

// ============================================================
// Stacked Expert Weights
// ============================================================

void StackedExpertWeights::build_from_experts(
    const std::vector<MoEWeights::ExpertW>& experts,
    uint32_t hidden, uint32_t intermediate)
{
    num_experts = experts.size();
    intermediate_size = intermediate;
    hidden_size = hidden;
    
    uint32_t mid = 2 * intermediate;
    gate_up_stacked.resize((size_t)num_experts * mid * hidden);
    down_stacked.resize((size_t)num_experts * hidden * intermediate);
    
    for (uint32_t e = 0; e < num_experts; e++) {
        memcpy(gate_up_stacked.data() + (size_t)e * mid * hidden,
               experts[e].gate_up.data(), mid * hidden * sizeof(float));
        memcpy(down_stacked.data() + (size_t)e * hidden * intermediate,
               experts[e].down.data(), hidden * intermediate * sizeof(float));
    }
}

// ============================================================
// Extended Decoder Layer
// ============================================================

void decoder_layer_extended_f32(
    const float* hidden_states,
    const DecoderLayerWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const DecoderLayerConfig& layer_cfg,
    const int* positions,
    float* output,
    float* debug_attn_out,
    float* debug_moe_out)
{
    uint32_t H = cfg.hidden_size;
    uint32_t total = n_tokens * H;
    
    // --- input_layernorm ---
    float* normed = new float[total];
    rms_norm_f32(hidden_states, w.input_ln_weight.data(), normed, n_tokens, H, cfg.rms_norm_eps);
    
    // --- self-attention ---
    float* attn_out = new float[total];
    
    if (layer_cfg.use_kv_cache && layer_cfg.kv_cache) {
        // KV-cache mode: single-token attention
        // QKV projection for the new token
        uint32_t n_heads = cfg.num_attention_heads;
        uint32_t n_kv_heads = cfg.num_key_value_heads;
        uint32_t head_dim = cfg.attention_head_dim;
        uint32_t qkv_total = cfg.qkv_total;
        
        float* qkv = new float[n_tokens * qkv_total];
        for (uint32_t i = 0; i < n_tokens; i++)
            for (uint32_t j = 0; j < qkv_total; j++) {
                float s = 0;
                for (uint32_t k = 0; k < H; k++)
                    s += normed[i * H + k] * w.attn.qkv_weight[j * H + k];
                qkv[i * qkv_total + j] = s;
            }
        
        // Split Q,K,V and store K,V in cache
        uint32_t hidden_q = cfg.hidden_size_q;
        uint32_t hidden_kv = cfg.hidden_size_kv;
        
        float* q = new float[n_tokens * n_heads * head_dim];
        float* k_new = new float[n_tokens * n_kv_heads * head_dim];
        float* v_new = new float[n_tokens * n_kv_heads * head_dim];
        
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < n_heads; h++)
                memcpy(q + (t * n_heads + h) * head_dim,
                       qkv + t * qkv_total + h * head_dim, head_dim * sizeof(float));
            for (uint32_t h = 0; h < n_kv_heads; h++) {
                memcpy(k_new + (t * n_kv_heads + h) * head_dim,
                       qkv + t * qkv_total + hidden_q + h * head_dim, head_dim * sizeof(float));
                memcpy(v_new + (t * n_kv_heads + h) * head_dim,
                       qkv + t * qkv_total + hidden_q + hidden_kv + h * head_dim, head_dim * sizeof(float));
            }
        }
        delete[] qkv;
        
        // Apply 2D RoPE if enabled
        if (layer_cfg.use_2d_rope && layer_cfg.rope_cos) {
            apply_rope_2d_f32(q, k_new, layer_cfg.rope_cos, layer_cfg.rope_sin,
                              n_tokens, n_heads, n_kv_heads, head_dim);
        }
        
        // QK Norm
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < n_heads; h++) {
                float* qh = q + (t * n_heads + h) * head_dim;
                float sum_sq = 0;
                for (uint32_t d = 0; d < head_dim; d++) sum_sq += qh[d] * qh[d];
                float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
                for (uint32_t d = 0; d < head_dim; d++) qh[d] *= rms * w.attn.q_norm_weight[d];
            }
            for (uint32_t h = 0; h < n_kv_heads; h++) {
                float* kh = k_new + (t * n_kv_heads + h) * head_dim;
                float sum_sq = 0;
                for (uint32_t d = 0; d < head_dim; d++) sum_sq += kh[d] * kh[d];
                float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
                for (uint32_t d = 0; d < head_dim; d++) kh[d] *= rms * w.attn.k_norm_weight[d];
            }
        }
        
        // Store K,V in cache
        KVCache* cache = layer_cfg.kv_cache;
        uint32_t pos = cache->current_len;
        for (uint32_t t = 0; t < n_tokens && pos + t < cache->max_seq_len; t++) {
            for (uint32_t h = 0; h < n_kv_heads; h++) {
                memcpy(cache->key_caches[layer_cfg.layer_idx] + 
                       (h * cache->max_seq_len + pos + t) * head_dim,
                       k_new + (t * n_kv_heads + h) * head_dim, head_dim * sizeof(float));
                memcpy(cache->value_caches[layer_cfg.layer_idx] + 
                       (h * cache->max_seq_len + pos + t) * head_dim,
                       v_new + (t * n_kv_heads + h) * head_dim, head_dim * sizeof(float));
            }
        }
        delete[] k_new; delete[] v_new;
        
        // Attention with cache
        float scale = 1.0f / sqrtf((float)head_dim);
        uint32_t seq_len = cache->current_len + n_tokens;
        
        for (uint32_t t = 0; t < n_tokens; t++) {
            float* out_t = attn_out + t * H;
            memset(out_t, 0, H * sizeof(float));
            
            for (uint32_t h = 0; h < n_heads; h++) {
                uint32_t kv_h = h / cfg.num_kv_groups;
                float* qh = q + (t * n_heads + h) * head_dim;
                
                // Scores
                float* scores = new float[seq_len];
                float max_s = -1e30f;
                for (uint32_t sk = 0; sk < seq_len; sk++) {
                    float dot = 0;
                    const float* kh = cache->key_caches[layer_cfg.layer_idx] + 
                        (kv_h * cache->max_seq_len + sk) * head_dim;
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
                    const float* vh = cache->value_caches[layer_cfg.layer_idx] + 
                        (kv_h * cache->max_seq_len + sk) * head_dim;
                    float sw = scores[sk];
                    for (uint32_t d = 0; d < head_dim; d++)
                        out_t[h * head_dim + d] += sw * vh[d];
                }
                delete[] scores;
            }
        }
        delete[] q;
        
        // Output projection
        float* merged = new float[n_tokens * H];
        for (uint32_t i = 0; i < n_tokens; i++)
            for (uint32_t j = 0; j < H; j++) {
                float s = 0;
                for (uint32_t k = 0; k < H; k++)
                    s += attn_out[i * H + k] * w.attn.o_weight[j * H + k];
                merged[i * H + j] = s;
            }
        memcpy(attn_out, merged, n_tokens * H * sizeof(float));
        delete[] merged;
        
        cache->current_len += n_tokens;
    } else {
        // Standard attention (no cache)
        attention_forward_cpu(normed, w.attn, n_tokens, cfg, positions, attn_out,
                              nullptr, nullptr, nullptr);
    }
    
    if (debug_attn_out) memcpy(debug_attn_out, attn_out, total * sizeof(float));
    
    // --- residual 1 ---
    float* h = new float[total];
    for (uint32_t i = 0; i < total; i++) h[i] = hidden_states[i] + attn_out[i];
    delete[] attn_out;
    
    // --- post_attention_layernorm ---
    rms_norm_f32(h, w.post_attn_ln_weight.data(), normed, n_tokens, H, cfg.rms_norm_eps);
    
    // --- MoE ---
    float* moe_out = new float[total];
    
    if (layer_cfg.routing_mode == DecoderLayerConfig::FULL) {
        // Full routing with all features
        StackedExpertWeights stacked;
        stacked.build_from_experts(w.moe.experts, H, cfg.moe_intermediate_size);
        
        moe_forward_full(normed, n_tokens, H,
                         w.moe.router_weight.data(),
                         w.moe.shared_gate_up.data(), w.moe.shared_down.data(),
                         stacked.gate_up_stacked.data(), stacked.down_stacked.data(),
                         w.moe.num_experts, cfg.moe_intermediate_size,
                         layer_cfg.moe_cfg, moe_out);
    } else {
        // Easy routing
        moe_forward_cpu(normed, w.moe, n_tokens, cfg, moe_out, nullptr, nullptr, nullptr);
    }
    
    if (debug_moe_out) memcpy(debug_moe_out, moe_out, total * sizeof(float));
    delete[] normed;
    
    // --- residual 2 ---
    for (uint32_t i = 0; i < total; i++) output[i] = h[i] + moe_out[i];
    delete[] h;
    delete[] moe_out;
}
