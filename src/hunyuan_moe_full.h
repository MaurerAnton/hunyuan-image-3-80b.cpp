#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================
// Full MoE Routing (DeepSeek-style topkgating)
//
// Supports ALL routing features:
//   - group_limited_greedy: group-based expert pre-selection
//   - capacity_factor: limit tokens per expert
//   - drop_tokens / random_routing_dropped_token
//   - routed_scaling_factor
//   - norm_topk_prob
//   - dispatch_mask construction + einsum combine
//   - auxiliary load balancing loss
//
// All features are runtime-configurable via MoERoutingConfig.
// When all features are disabled, behavior matches easy_topk.
// ============================================================

struct MoERoutingConfig {
    // Basic
    uint32_t num_experts = 64;
    uint32_t moe_topk = 8;
    
    // Group-limited greedy (DeepSeek-V2 style)
    bool group_limited_greedy = false;
    uint32_t n_group = 0;        // number of expert groups
    uint32_t topk_group = 0;     // top-K groups to select
    
    // Capacity control
    bool drop_tokens = false;
    bool random_routing_dropped_token = false;
    float capacity_factor = 1.0f;
    uint32_t min_capacity = 8;
    
    // Probability handling
    bool norm_topk_prob = true;
    float routed_scaling_factor = 1.0f;
    
    // Aux loss
    bool compute_aux_loss = false;
};

struct MoEGateOutput {
    // For default mode: dispatch_mask routing
    float* combine_weights = nullptr;   // [tokens, experts, capacity]
    bool*  dispatch_mask = nullptr;     // [tokens, experts, capacity]
    int32_t* exp_counts = nullptr;       // [num_experts]
    float aux_loss = 0.0f;
    float exp_capacity_rate = 0.0f;
    uint32_t expert_capacity = 0;
    
    // For easy mode / direct routing
    float* topk_weights = nullptr;      // [tokens, topk]
    int32_t* topk_indices = nullptr;    // [tokens, topk]
    float* router_probs = nullptr;      // [tokens, experts] — full softmax
    
    uint32_t n_tokens = 0;
    bool use_dispatch_mask = false;
    
    ~MoEGateOutput();
    
    // Allocate for a given number of tokens
    void alloc(uint32_t tokens, uint32_t experts, uint32_t topk, uint32_t capacity, bool dispatch);
};

// ============================================================
// Full topkgating (DeepSeek-V2 compatible)
// ============================================================

void topkgating_full(
    const float* logits,           // [n_tokens, num_experts]
    uint32_t n_tokens,
    const MoERoutingConfig& cfg,
    MoEGateOutput& out);

// Simplified: softmax → top-k → normalize (equivalent to easy_topk)
void topkgating_easy(
    const float* logits,
    uint32_t n_tokens,
    uint32_t num_experts,
    uint32_t topk,
    float* topk_weights,          // [n_tokens, topk]
    int32_t* topk_indices);       // [n_tokens, topk]

// ============================================================
// Einsum dispatch: input → per-expert chunks
// ============================================================

void einsum_dispatch(
    const float* input,            // [n_tokens, hidden]
    uint32_t n_tokens,
    uint32_t hidden_size,
    const bool* dispatch_mask,    // [n_tokens, experts, capacity]
    uint32_t num_experts,
    uint32_t expert_capacity,
    float* dispatched_input);     // [experts, capacity, hidden]

// ============================================================
// Einsum combine: per-expert outputs → combined output
// ============================================================

void einsum_combine(
    const float* combine_weights,  // [n_tokens, experts, capacity]
    const float* expert_output,    // [experts, capacity, hidden]
    uint32_t n_tokens,
    uint32_t hidden_size,
    uint32_t num_experts,
    uint32_t expert_capacity,
    float* combined_output);      // [n_tokens, hidden]

// ============================================================
// Full MoE forward (default mode with all features)
// ============================================================

void moe_forward_full(
    const float* hidden_states,    // [n_tokens, hidden_size]
    uint32_t n_tokens,
    uint32_t hidden_size,
    
    // Router weight: [num_experts, hidden_size]
    const float* router_weight,
    
    // Shared expert weights (optional, can be nullptr for no shared)
    const float* shared_gate_up,   // [2*intermediate, hidden_size]
    const float* shared_down,      // [hidden_size, intermediate]
    
    // Expert weights: stacked [num_experts, 2*intermediate, hidden_size]
    // and [num_experts, hidden_size, intermediate]
    const float* expert_gate_up_stacked,  // [num_experts * 2*intermediate, hidden_size]
    const float* expert_down_stacked,     // [num_experts * hidden_size, intermediate]
    
    uint32_t num_experts,
    uint32_t intermediate_size,
    const MoERoutingConfig& routing_cfg,
    
    float* output);               // [n_tokens, hidden_size]

// ============================================================
// Softmax (safe, with max subtraction)
// ============================================================

void softmax_f32(float* x, uint32_t n);

// ============================================================
// Top-K (returns indices sorted by value descending)
// ============================================================

void topk_f32(const float* x, uint32_t n, uint32_t k,
              float* topk_vals, int32_t* topk_idx);
