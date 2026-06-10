#pragma once

#include "hunyuan_model.h"
#include "hunyuan_image_gen.h"
#include "hunyuan_moe_full.h"
#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================
// Extended Decoder Layer with all optional features
//
// Supports:
//   - KV cache for autoregressive generation
//   - Easy vs Full MoE routing (runtime switch)
//   - 2D RoPE (via precomputed cos/sin)
//   - Output hidden states (for debugging)
// ============================================================

struct DecoderLayerConfig {
    // Routing mode
    enum RoutingMode { EASY, FULL, FLASHINFER };
    RoutingMode routing_mode = EASY;
    MoERoutingConfig moe_cfg;  // Used when routing_mode == FULL
    
    // Attention
    bool use_2d_rope = false;
    const float* rope_cos = nullptr;  // [seq_len, head_dim]
    const float* rope_sin = nullptr;
    
    // KV Cache
    bool use_kv_cache = false;
    KVCache* kv_cache = nullptr;
    uint32_t layer_idx = 0;
};

void decoder_layer_extended_f32(
    const float* hidden_states,    // [n_tokens, hidden_size]
    const DecoderLayerWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const DecoderLayerConfig& layer_cfg,
    const int* positions,
    float* output,                 // [n_tokens, hidden_size]
    // Optional debug outputs
    float* debug_attn_out = nullptr,
    float* debug_moe_out = nullptr);

// ============================================================
// Stacked expert weights (for FlashInfer-compatible layout)
// ============================================================

struct StackedExpertWeights {
    std::vector<float> gate_up_stacked;  // [num_experts * 2*intermediate, hidden]
    std::vector<float> down_stacked;     // [num_experts * hidden, intermediate]
    uint32_t num_experts = 0;
    uint32_t intermediate_size = 0;
    uint32_t hidden_size = 0;
    
    // Build from per-expert weight vectors
    void build_from_experts(const std::vector<MoEWeights::ExpertW>& experts,
                            uint32_t hidden, uint32_t intermediate);
    
    // Release individual expert weights and keep only stacked
    void release_per_expert_weights();
};
