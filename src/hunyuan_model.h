#pragma once

#include "ggml.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

// ============================================================
// Configuration (mirrors config.json)
// ============================================================
struct HunyuanConfig {
    uint32_t hidden_size = 4096;
    uint32_t num_hidden_layers = 32;
    uint32_t num_attention_heads = 32;
    uint32_t num_key_value_heads = 8;
    uint32_t attention_head_dim = 128;
    uint32_t intermediate_size = 3072;
    uint32_t moe_intermediate_size = 3072;
    uint32_t num_experts = 64;
    uint32_t moe_topk = 8;
    uint32_t num_shared_expert = 1;
    uint32_t vocab_size = 133120;
    uint32_t max_position_embeddings = 22800;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool use_qk_norm = true;
    bool use_mixed_mlp_moe = true;
    
    // Derived
    uint32_t num_kv_groups;
    uint32_t hidden_size_q;   // num_heads * head_dim
    uint32_t hidden_size_kv;  // num_kv_heads * head_dim
    uint32_t qkv_total;       // hidden_size_q + 2 * hidden_size_kv
    
    void compute_derived() {
        num_kv_groups = num_attention_heads / num_key_value_heads;
        hidden_size_q = num_attention_heads * attention_head_dim;
        hidden_size_kv = num_key_value_heads * attention_head_dim;
        qkv_total = hidden_size_q + 2 * hidden_size_kv;
    }
    
    static HunyuanConfig from_json(const std::string& path);
};

// ============================================================
// Tensor storage (for layer-by-layer: we load tensors individually)
// ============================================================
struct ModelTensor {
    std::string name;
    std::vector<uint32_t> shape;  // ne[0..3]
    std::vector<float> data;       // stored as f32
    ggml_type dtype;
    
    void load_from_safetensors(const std::string& filepath, const std::string& tensor_name);
};

// ============================================================
// RMS Normalization
// ============================================================
void rms_norm_f32(
    const float* x,      // [n_tokens, hidden_size]
    const float* weight, // [hidden_size]
    float* out,          // [n_tokens, hidden_size]
    uint32_t n_tokens,
    uint32_t hidden_size,
    float eps);

// GGML graph version
ggml_tensor* rms_norm_ggml(
    ggml_context* ctx,
    ggml_tensor* x,      // [hidden_size, n_tokens]
    ggml_tensor* weight, // [hidden_size]
    float eps);

// ============================================================
// Rotary Position Embedding (1D for now, 2D later)
// ============================================================
void rope_f32(
    float* q,             // [seq_len, n_heads, head_dim] — in-place
    float* k,             // [seq_len, n_kv_heads, head_dim] — in-place
    uint32_t seq_len,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta,
    const int* positions); // [seq_len]

// ============================================================
// Attention
// ============================================================
struct AttentionWeights {
    // qkv_proj: [qkv_total, hidden_size] — fused QKV
    std::vector<float> qkv_weight;  // [qkv_total, hidden_size]
    // o_proj: [hidden_size, hidden_size_q]
    std::vector<float> o_weight;    // [hidden_size, hidden_size_q]
    // QK norm: [head_dim]
    std::vector<float> q_norm_weight;
    std::vector<float> k_norm_weight;
};

void attention_forward_cpu(
    const float* hidden_states,   // [n_tokens, hidden_size]
    const AttentionWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const int* positions,          // [n_tokens]
    float* output,                 // [n_tokens, hidden_size]
    // Debug outputs (optional)
    float* debug_q = nullptr,      // [n_tokens, n_heads, head_dim]
    float* debug_k = nullptr,      // [n_tokens, n_kv_heads, head_dim]
    float* debug_v = nullptr);     // [n_tokens, n_kv_heads, head_dim]

// ============================================================
// SwiGLU FFN (single expert)
// ============================================================
void swiglu_ffn_f32(
    const float* x,              // [n, hidden_size]
    const float* gate_up_weight, // [2*intermediate, hidden_size]
    const float* down_weight,    // [hidden_size, intermediate]
    float* out,                  // [n, hidden_size]
    uint32_t n,
    uint32_t hidden_size,
    uint32_t intermediate_size);

// ============================================================
// MoE (Mixture of Experts)
// ============================================================
struct MoEWeights {
    // Shared expert
    std::vector<float> shared_gate_up;  // [2*intermediate, hidden_size]
    std::vector<float> shared_down;      // [hidden_size, intermediate]
    
    // Router: [num_experts, hidden_size]
    std::vector<float> router_weight;    // [num_experts, hidden_size]
    
    // 64 routed experts
    struct ExpertW {
        std::vector<float> gate_up;  // [2*intermediate, hidden_size]
        std::vector<float> down;     // [hidden_size, intermediate]
    };
    std::vector<ExpertW> experts;
    uint32_t num_experts;
};

void moe_forward_cpu(
    const float* hidden_states,  // [n_tokens, hidden_size]
    const MoEWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    float* output,               // [n_tokens, hidden_size]
    // Debug outputs
    float* debug_router_logits = nullptr,   // [n_tokens, num_experts]
    float* debug_router_weights = nullptr,  // [n_tokens, topk]
    int32_t* debug_router_indices = nullptr); // [n_tokens, topk]

// ============================================================
// Decoder Layer
// ============================================================
struct DecoderLayerWeights {
    // Norm weights
    std::vector<float> input_ln_weight;      // [hidden_size]
    std::vector<float> post_attn_ln_weight;  // [hidden_size]
    
    // Attention
    AttentionWeights attn;
    
    // MoE
    MoEWeights moe;
    
    // Load from safetensors shard
    bool load(int layer_idx, const std::string& shard_path);
};

void decoder_layer_forward_cpu(
    const float* hidden_states,   // [n_tokens, hidden_size]
    const DecoderLayerWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const int* positions,
    float* output);               // [n_tokens, hidden_size]

// ============================================================
// Helpers
// ============================================================
float tensor_rms(const float* x, uint32_t n);
float tensor_correlation(const float* a, const float* b, uint32_t n);
void print_tensor_stats(const char* label, const float* x, uint32_t n);
