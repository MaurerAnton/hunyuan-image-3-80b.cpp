#include "hunyuan_model.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

// ============================================================
// Attention Forward (pure C++ on CPU)
// ============================================================

void attention_forward_cpu(
    const float* hidden_states,
    const AttentionWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const int* positions,
    float* output,
    float* debug_q,
    float* debug_k,
    float* debug_v)
{
    uint32_t H = cfg.hidden_size;         // 4096
    uint32_t n_heads = cfg.num_attention_heads;     // 32
    uint32_t n_kv_heads = cfg.num_key_value_heads;  // 8
    uint32_t head_dim = cfg.attention_head_dim;     // 128
    uint32_t n_groups = cfg.num_kv_groups;           // 4
    uint32_t qkv_total = cfg.qkv_total;             // 6144
    
    // Step 1: QKV projection
    // hidden_states: [n_tokens, H]
    // qkv_weight: [qkv_total, H]
    // result: [n_tokens, qkv_total]
    float* qkv = new float[n_tokens * qkv_total];
    for (uint32_t i = 0; i < n_tokens; i++) {
        for (uint32_t j = 0; j < qkv_total; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < H; k++) {
                s += hidden_states[i * H + k] * w.qkv_weight[j * H + k];
            }
            qkv[i * qkv_total + j] = s;
        }
    }
    
    // Step 2: Split QKV and reshape to [tokens, heads, head_dim]
    uint32_t hidden_q  = cfg.hidden_size_q;   // 4096 = 32*128
    uint32_t hidden_kv = cfg.hidden_size_kv;  // 1024 = 8*128
    
    // Q: [n_tokens, n_heads, head_dim] — layout: token-major, then head, then dim
    //    element [t][h][d] = q[t * n_heads * head_dim + h * head_dim + d]
    float* q = new float[n_tokens * n_heads * head_dim];
    float* k = new float[n_tokens * n_kv_heads * head_dim];
    float* v = new float[n_tokens * n_kv_heads * head_dim];
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        // Q: heads are contiguous groups of head_dim from qkv
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                q[(t * n_heads + h) * head_dim + d] = 
                    qkv[t * qkv_total + h * head_dim + d];
            }
        }
        // K: follows Q
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                k[(t * n_kv_heads + h) * head_dim + d] = 
                    qkv[t * qkv_total + hidden_q + h * head_dim + d];
            }
        }
        // V: follows K
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                v[(t * n_kv_heads + h) * head_dim + d] = 
                    qkv[t * qkv_total + hidden_q + hidden_kv + h * head_dim + d];
            }
        }
    }
    delete[] qkv;
    
    // Step 3: Apply RoPE to Q and K
    rope_f32(q, k, n_tokens, n_heads, n_kv_heads, head_dim, cfg.rope_theta, positions);
    
    // Step 4: QK Normalization (RMSNorm per head)
    // Q norm: normalize each head_dim vector
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            float* qh = q + (t * n_heads + h) * head_dim;
            // RMS
            float sum_sq = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) sum_sq += qh[d] * qh[d];
            float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
            for (uint32_t d = 0; d < head_dim; d++) qh[d] = qh[d] * rms * w.q_norm_weight[d];
        }
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            float* kh = k + (t * n_kv_heads + h) * head_dim;
            float sum_sq = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) sum_sq += kh[d] * kh[d];
            float rms = 1.0f / sqrtf(sum_sq / head_dim + cfg.rms_norm_eps);
            for (uint32_t d = 0; d < head_dim; d++) kh[d] = kh[d] * rms * w.k_norm_weight[d];
        }
    }
    
    // Save debug Q, K, V
    if (debug_q) memcpy(debug_q, q, n_tokens * n_heads * head_dim * sizeof(float));
    if (debug_k) memcpy(debug_k, k, n_tokens * n_kv_heads * head_dim * sizeof(float));
    if (debug_v) memcpy(debug_v, v, n_tokens * n_kv_heads * head_dim * sizeof(float));
    
    // Step 5: Scaled Dot-Product Attention (per head, causal mask)
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Output: [n_tokens, n_heads, head_dim]
    float* attn_out = new float[n_tokens * n_heads * head_dim]();
    
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = h / n_groups;  // which KV head this Q head maps to
        
        // For each query position
        for (uint32_t t_q = 0; t_q < n_tokens; t_q++) {
            // Compute attention scores for all keys up to t_q (causal)
            float* scores = new float[t_q + 1];
            float max_score = -1e30f;
            
            for (uint32_t t_k = 0; t_k <= t_q; t_k++) {
                float dot = 0.0f;
                const float* qh = q + (t_q * n_heads + h) * head_dim;
                const float* kh = k + (t_k * n_kv_heads + kv_h) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += qh[d] * kh[d];
                }
                scores[t_k] = dot * scale;
                if (scores[t_k] > max_score) max_score = scores[t_k];
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (uint32_t t_k = 0; t_k <= t_q; t_k++) {
                scores[t_k] = expf(scores[t_k] - max_score);
                sum_exp += scores[t_k];
            }
            for (uint32_t t_k = 0; t_k <= t_q; t_k++) {
                scores[t_k] /= sum_exp;
            }
            
            // Weighted sum of values
            float* out_h = attn_out + (t_q * n_heads + h) * head_dim;
            memset(out_h, 0, head_dim * sizeof(float));
            for (uint32_t t_k = 0; t_k <= t_q; t_k++) {
                const float* vh = v + (t_k * n_kv_heads + kv_h) * head_dim;
                float w = scores[t_k];
                for (uint32_t d = 0; d < head_dim; d++) {
                    out_h[d] += w * vh[d];
                }
            }
            
            delete[] scores;
        }
    }
    
    delete[] q;
    delete[] k;
    delete[] v;
    
    // Step 6: Merge heads and output projection
    // attn_out: [n_tokens, n_heads, head_dim] → flatten to [n_tokens, n_heads*head_dim]
    float* merged = new float[n_tokens * hidden_q];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            memcpy(merged + t * hidden_q + h * head_dim,
                   attn_out + (t * n_heads + h) * head_dim,
                   head_dim * sizeof(float));
        }
    }
    delete[] attn_out;
    
    // o_proj: [hidden_size, hidden_q]
    // output = merged @ o_weight^T = [n_tokens, hidden_q] @ [hidden_q, hidden_size]
    for (uint32_t i = 0; i < n_tokens; i++) {
        for (uint32_t j = 0; j < H; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < hidden_q; k++) {
                s += merged[i * hidden_q + k] * w.o_weight[j * hidden_q + k];
            }
            output[i * H + j] = s;
        }
    }
    delete[] merged;
}

// ============================================================
// MoE Forward
// ============================================================

void moe_forward_cpu(
    const float* hidden_states,
    const MoEWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    float* output,
    float* debug_router_logits,
    float* debug_router_weights,
    int32_t* debug_router_indices)
{
    uint32_t H = cfg.hidden_size;
    uint32_t intermediate = cfg.moe_intermediate_size;  // 3072
    uint32_t n_experts = w.num_experts;                   // 64
    uint32_t topk = cfg.moe_topk;                          // 8
    
    // Step 1: Shared expert
    float* shared_out = new float[n_tokens * H];
    swiglu_ffn_f32(hidden_states, w.shared_gate_up.data(), w.shared_down.data(),
                   shared_out, n_tokens, H, intermediate);
    
    // Step 2: Router — compute logits and top-k
    // router_weight: [n_experts, H]
    float* router_logits = new float[n_tokens * n_experts];
    for (uint32_t i = 0; i < n_tokens; i++) {
        for (uint32_t e = 0; e < n_experts; e++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < H; k++) {
                s += hidden_states[i * H + k] * w.router_weight[e * H + k];
            }
            router_logits[i * n_experts + e] = s;
        }
    }
    
    // Softmax over experts
    for (uint32_t i = 0; i < n_tokens; i++) {
        float max_logit = router_logits[i * n_experts];
        for (uint32_t e = 1; e < n_experts; e++) {
            if (router_logits[i * n_experts + e] > max_logit)
                max_logit = router_logits[i * n_experts + e];
        }
        float sum_exp = 0.0f;
        for (uint32_t e = 0; e < n_experts; e++) {
            router_logits[i * n_experts + e] = 
                expf(router_logits[i * n_experts + e] - max_logit);
            sum_exp += router_logits[i * n_experts + e];
        }
        for (uint32_t e = 0; e < n_experts; e++) {
            router_logits[i * n_experts + e] /= sum_exp;
        }
    }
    
    // Top-k selection (simple linear scan with tracking)
    float* topk_weights = new float[n_tokens * topk];
    int32_t* topk_indices = new int32_t[n_tokens * topk];
    
    for (uint32_t i = 0; i < n_tokens; i++) {
        // Find top-k experts
        struct Pair { float w; int32_t idx; };
        Pair best[8];  // topk=8
        int n_best = 0;
        
        for (uint32_t e = 0; e < n_experts; e++) {
            float val = router_logits[i * n_experts + e];
            // Insert into sorted best list
            int pos = n_best;
            while (pos > 0 && best[pos-1].w < val) {
                if (pos < (int)topk) best[pos] = best[pos-1];
                pos--;
            }
            if (pos < (int)topk) {
                best[pos].w = val;
                best[pos].idx = e;
                if (n_best < (int)topk) n_best++;
            }
        }
        
        // Normalize top-k weights
        float sum_w = 0.0f;
        for (uint32_t k = 0; k < topk; k++) {
            sum_w += best[k].w;
        }
        for (uint32_t k = 0; k < topk; k++) {
            topk_weights[i * topk + k] = best[k].w / sum_w;
            topk_indices[i * topk + k] = best[k].idx;
        }
    }
    
    if (debug_router_logits) memcpy(debug_router_logits, router_logits, n_tokens * n_experts * sizeof(float));
    if (debug_router_weights) memcpy(debug_router_weights, topk_weights, n_tokens * topk * sizeof(float));
    if (debug_router_indices) memcpy(debug_router_indices, topk_indices, n_tokens * topk * sizeof(int32_t));
    
    // Step 3: Routed experts — compute weighted sum
    float* routed_out = new float[n_tokens * H]();
    
    for (uint32_t i = 0; i < n_tokens; i++) {
        for (uint32_t ki = 0; ki < topk; ki++) {
            int32_t expert_idx = topk_indices[i * topk + ki];
            float weight = topk_weights[i * topk + ki];
            
            const MoEWeights::ExpertW& expert = w.experts[expert_idx];
            float expert_out[H];
            swiglu_ffn_f32(hidden_states + i * H,
                          expert.gate_up.data(), expert.down.data(),
                          expert_out, 1, H, intermediate);
            
            for (uint32_t j = 0; j < H; j++) {
                routed_out[i * H + j] += weight * expert_out[j];
            }
        }
    }
    
    delete[] router_logits;
    delete[] topk_weights;
    delete[] topk_indices;
    
    // Step 4: Combine shared + routed
    for (uint32_t i = 0; i < n_tokens * H; i++) {
        output[i] = shared_out[i] + routed_out[i];
    }
    
    delete[] shared_out;
    delete[] routed_out;
}
