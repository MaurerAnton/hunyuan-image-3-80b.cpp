#include "hunyuan_moe_full.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================
// MoEGateOutput
// ============================================================

MoEGateOutput::~MoEGateOutput() {
    free(combine_weights);
    free(dispatch_mask);
    free(exp_counts);
    free(topk_weights);
    free(topk_indices);
    free(router_probs);
}

void MoEGateOutput::alloc(uint32_t tokens, uint32_t experts, uint32_t topk, 
                           uint32_t capacity, bool dispatch) {
    n_tokens = tokens;
    use_dispatch_mask = dispatch;
    
    if (dispatch) {
        size_t mask_size = (size_t)tokens * experts * capacity;
        dispatch_mask = (bool*)calloc(mask_size, sizeof(bool));
        combine_weights = (float*)calloc(mask_size, sizeof(float));
        exp_counts = (int32_t*)calloc(experts, sizeof(int32_t));
    } else {
        topk_weights = (float*)malloc(tokens * topk * sizeof(float));
        topk_indices = (int32_t*)malloc(tokens * topk * sizeof(int32_t));
    }
    router_probs = (float*)malloc(tokens * experts * sizeof(float));
}

// ============================================================
// Softmax
// ============================================================

void softmax_f32(float* x, uint32_t n) {
    float max_val = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (uint32_t i = 0; i < n; i++) x[i] /= sum;
}

// ============================================================
// Top-K
// ============================================================

void topk_f32(const float* x, uint32_t n, uint32_t k,
              float* topk_vals, int32_t* topk_idx) {
    // Simple insertion sort for the first k elements
    uint32_t n_best = 0;
    struct Pair { float v; int32_t i; } best[64]; // max k=64
    memset(best, 0, sizeof(best));
    
    for (uint32_t i = 0; i < n; i++) {
        float val = x[i];
        // Find insertion point
        uint32_t pos = n_best;
        while (pos > 0 && best[pos-1].v < val) pos--;
        // Shift and insert
        if (pos < k) {
            for (uint32_t j = std::min(n_best, k-1); j > pos; j--)
                best[j] = best[j-1];
            best[pos].v = val;
            best[pos].i = (int32_t)i;
            if (n_best < k) n_best++;
        }
    }
    
    for (uint32_t i = 0; i < k; i++) {
        topk_vals[i] = best[i].v;
        topk_idx[i] = best[i].i;
    }
}

// ============================================================
// Easy topkgating
// ============================================================

void topkgating_easy(
    const float* logits, uint32_t n_tokens,
    uint32_t num_experts, uint32_t topk,
    float* topk_weights, int32_t* topk_indices)
{
    for (uint32_t t = 0; t < n_tokens; t++) {
        // Softmax
        float* probs = new float[num_experts];
        memcpy(probs, logits + t * num_experts, num_experts * sizeof(float));
        softmax_f32(probs, num_experts);
        
        // Top-K
        float* vals = new float[topk];
        topk_f32(probs, num_experts, topk, vals, topk_indices + t * topk);
        
        // Normalize
        float sum = 0;
        for (uint32_t k = 0; k < topk; k++) sum += vals[k];
        if (sum < 1e-8f) sum = 1e-8f;
        for (uint32_t k = 0; k < topk; k++)
            topk_weights[t * topk + k] = vals[k] / sum;
        
        delete[] probs;
        delete[] vals;
    }
}

// ============================================================
// Full topkgating (DeepSeek-V2 compatible)
// ============================================================

void topkgating_full(
    const float* logits, uint32_t n_tokens,
    const MoERoutingConfig& cfg, MoEGateOutput& out)
{
    uint32_t E = cfg.num_experts;
    uint32_t K = cfg.moe_topk;
    
    // Step 1: Softmax over experts
    float* gates = new float[n_tokens * E];
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(gates + t * E, logits + t * E, E * sizeof(float));
        softmax_f32(gates + t * E, E);
    }
    memcpy(out.router_probs, gates, n_tokens * E * sizeof(float));
    
    // Step 2: Group-limited greedy (optional)
    if (cfg.group_limited_greedy && cfg.n_group > 0 && cfg.topk_group > 0) {
        uint32_t group_size = E / cfg.n_group;
        float* group_scores = new float[n_tokens * cfg.n_group];
        
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t g = 0; g < cfg.n_group; g++) {
                float max_s = 0;
                for (uint32_t e = 0; e < group_size; e++)
                    if (gates[t * E + g * group_size + e] > max_s)
                        max_s = gates[t * E + g * group_size + e];
                group_scores[t * cfg.n_group + g] = max_s;
            }
        }
        
        // Select top-k_groups
        float* gvals = new float[cfg.topk_group];
        int32_t* gidx = new int32_t[cfg.topk_group];
        
        for (uint32_t t = 0; t < n_tokens; t++) {
            topk_f32(group_scores + t * cfg.n_group, cfg.n_group, cfg.topk_group, gvals, gidx);
            // Zero out experts not in selected groups
            for (uint32_t g = 0; g < cfg.n_group; g++) {
                bool selected = false;
                for (uint32_t kg = 0; kg < cfg.topk_group; kg++)
                    if (gidx[kg] == (int32_t)g) { selected = true; break; }
                if (!selected) {
                    memset(gates + t * E + g * group_size, 0, group_size * sizeof(float));
                }
            }
        }
        delete[] group_scores; delete[] gvals; delete[] gidx;
    }
    
    // Step 3: Top-K per token
    float* tk_vals = new float[K];
    int32_t* tk_idx = new int32_t[K];
    float* expert_gate = new float[n_tokens * K];
    int32_t* expert_index = new int32_t[n_tokens * K];
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        topk_f32(gates + t * E, E, K, tk_vals, tk_idx);
        memcpy(expert_gate + t * K, tk_vals, K * sizeof(float));
        memcpy(expert_index + t * K, tk_idx, K * sizeof(int32_t));
    }
    
    // Step 4: Expert mask and load balancing aux loss
    float l_aux = 0.0f;
    if (cfg.compute_aux_loss) {
        // expert_mask: [tokens, experts] — which tokens go to which expert
        float* expert_mask_aux = new float[n_tokens * E]();
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t k = 0; k < K; k++)
                expert_mask_aux[t * E + expert_index[t * K + k]] = 1.0f;
        
        float* tokens_per_expert = new float[E]();
        float* router_prob_per_expert = new float[E]();
        for (uint32_t e = 0; e < E; e++) {
            for (uint32_t t = 0; t < n_tokens; t++) {
                tokens_per_expert[e] += expert_mask_aux[t * E + e];
                router_prob_per_expert[e] += gates[t * E + e];
            }
            tokens_per_expert[e] /= n_tokens;
            router_prob_per_expert[e] /= n_tokens;
        }
        for (uint32_t e = 0; e < E; e++)
            l_aux += tokens_per_expert[e] * router_prob_per_expert[e];
        l_aux *= E * E;
        
        delete[] expert_mask_aux;
        delete[] tokens_per_expert;
        delete[] router_prob_per_expert;
    }
    out.aux_loss = l_aux;
    
    // Step 5: Capacity calculation
    uint32_t expert_capacity;
    if (cfg.drop_tokens) {
        expert_capacity = (uint32_t)(std::max((uint32_t)K, 
            (uint32_t)(cfg.capacity_factor * K * n_tokens / E)));
        if (expert_capacity < cfg.min_capacity) expert_capacity = cfg.min_capacity;
    } else {
        // Count tokens per expert
        uint32_t* counts = new uint32_t[E]();
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t k = 0; k < K; k++)
                counts[expert_index[t * K + k]]++;
        expert_capacity = 0;
        for (uint32_t e = 0; e < E; e++)
            if (counts[e] > expert_capacity) expert_capacity = counts[e];
        if (expert_capacity < cfg.min_capacity) expert_capacity = cfg.min_capacity;
        delete[] counts;
    }
    out.expert_capacity = expert_capacity;
    
    // Step 6: Router probabilities (norm_topk_prob or scaling)
    if (cfg.norm_topk_prob && K > 1) {
        // expert_mask summed: sum of gates for each token across selected experts
        for (uint32_t t = 0; t < n_tokens; t++) {
            float sum_g = 0;
            for (uint32_t k = 0; k < K; k++) sum_g += gates[t * E + expert_index[t * K + k]];
            if (sum_g < 1e-8f) sum_g = 1e-8f;
            for (uint32_t k = 0; k < K; k++)
                expert_gate[t * K + k] = gates[t * E + expert_index[t * K + k]] / sum_g;
        }
    } else {
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t k = 0; k < K; k++)
                expert_gate[t * K + k] = gates[t * E + expert_index[t * K + k]] * cfg.routed_scaling_factor;
    }
    
    // Step 7: Dispatch mask construction
    // expert_index transposed: [K, tokens] → flatten
    // This prioritizes top-1 over top-2, etc.
    int32_t* expert_index_flat = new int32_t[n_tokens * K];
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t t = 0; t < n_tokens; t++)
            expert_index_flat[k * n_tokens + t] = expert_index[t * K + k];
    
    // expert_mask: one-hot [tokens*K, num_experts]
    uint8_t* expert_mask_oh = new uint8_t[(size_t)n_tokens * K * E]();
    for (uint32_t i = 0; i < n_tokens * K; i++)
        expert_mask_oh[i * E + expert_index_flat[i]] = 1;
    
    // exp_counts
    memset(out.exp_counts, 0, E * sizeof(int32_t));
    for (uint32_t i = 0; i < n_tokens * K; i++)
        out.exp_counts[expert_index_flat[i]]++;
    
    // token_priority: cumulative sum of expert_mask
    int32_t* token_priority = new int32_t[(size_t)n_tokens * K * E];
    for (uint32_t e = 0; e < E; e++) {
        int32_t cumsum = 0;
        for (uint32_t i = 0; i < n_tokens * K; i++) {
            if (expert_mask_oh[i * E + e]) {
                token_priority[i * E + e] = cumsum++;
            } else {
                token_priority[i * E + e] = -1;
            }
        }
    }
    
    // Reshape: [n_tokens*K, E] → [K, n_tokens, E] → transpose → [n_tokens, K, E]
    // Then max over K → [n_tokens, E]
    int32_t* tp_reshaped = new int32_t[(size_t)n_tokens * K * E];
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t e = 0; e < E; e++)
                tp_reshaped[(t * K + k) * E + e] = token_priority[(k * n_tokens + t) * E + e];
    
    int32_t* tp_max = new int32_t[n_tokens * E];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < E; e++) {
            int32_t max_p = -1;
            for (uint32_t k = 0; k < K; k++)
                if (tp_reshaped[(t * K + k) * E + e] > max_p)
                    max_p = tp_reshaped[(t * K + k) * E + e];
            tp_max[t * E + e] = max_p;
        }
    }
    
    // Build dispatch_mask: valid if 0 <= priority < expert_capacity
    memset(out.dispatch_mask, 0, (size_t)n_tokens * E * expert_capacity * sizeof(bool));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < E; e++) {
            int32_t p = tp_max[t * E + e];
            if (p >= 0 && (uint32_t)p < expert_capacity) {
                out.dispatch_mask[(t * E + e) * expert_capacity + p] = true;
            }
        }
    }
    
    // Build combine_weights: router_probs * dispatch_mask
    // router_probs are from expert_gate, broadcast to [n_tokens, K, E]
    // combine = einsum("te,tec->tec", router_probs, dispatch_mask)
    memset(out.combine_weights, 0, (size_t)n_tokens * E * expert_capacity * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < E; e++) {
            // Find which top-k slot this expert was selected for
            float prob = 0;
            for (uint32_t k = 0; k < K; k++) {
                if (expert_index[t * K + k] == (int32_t)e) {
                    prob = expert_gate[t * K + k];
                    break;
                }
            }
            for (uint32_t c = 0; c < expert_capacity; c++) {
                if (out.dispatch_mask[(t * E + e) * expert_capacity + c]) {
                    out.combine_weights[(t * E + e) * expert_capacity + c] = prob;
                }
            }
        }
    }
    
    // Capacity rate
    float exp_counts_capacity = 0;
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t e = 0; e < E; e++)
            for (uint32_t c = 0; c < expert_capacity; c++)
                if (out.dispatch_mask[(t * E + e) * expert_capacity + c])
                    exp_counts_capacity += 1;
    out.exp_capacity_rate = exp_counts_capacity / (n_tokens * K);
    
    // Cleanup
    delete[] gates;
    delete[] tk_vals; delete[] tk_idx;
    delete[] expert_gate; delete[] expert_index;
    delete[] expert_index_flat;
    delete[] expert_mask_oh;
    delete[] token_priority;
    delete[] tp_reshaped;
    delete[] tp_max;
    delete[] gates;
}

// ============================================================
// Einsum dispatch
// ============================================================

void einsum_dispatch(
    const float* input, uint32_t n_tokens, uint32_t hidden,
    const bool* dispatch_mask, uint32_t E, uint32_t capacity,
    float* dispatched_input)
{
    // dispatched_input: [E, capacity, hidden]
    size_t out_size = (size_t)E * capacity * hidden;
    memset(dispatched_input, 0, out_size * sizeof(float));
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < E; e++) {
            for (uint32_t c = 0; c < capacity; c++) {
                if (dispatch_mask[(t * E + e) * capacity + c]) {
                    for (uint32_t h = 0; h < hidden; h++) {
                        dispatched_input[(e * capacity + c) * hidden + h] += 
                            input[t * hidden + h];
                    }
                }
            }
        }
    }
}

// ============================================================
// Einsum combine
// ============================================================

void einsum_combine(
    const float* combine_weights, const float* expert_output,
    uint32_t n_tokens, uint32_t hidden,
    uint32_t E, uint32_t capacity,
    float* combined_output)
{
    memset(combined_output, 0, (size_t)n_tokens * hidden * sizeof(float));
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < E; e++) {
            for (uint32_t c = 0; c < capacity; c++) {
                float w = combine_weights[(t * E + e) * capacity + c];
                if (w == 0) continue;
                for (uint32_t h = 0; h < hidden; h++) {
                    combined_output[t * hidden + h] += 
                        w * expert_output[(e * capacity + c) * hidden + h];
                }
            }
        }
    }
}

// ============================================================
// Full MoE forward
// ============================================================

void moe_forward_full(
    const float* hidden_states, uint32_t n_tokens, uint32_t hidden,
    const float* router_weight,
    const float* shared_gate_up, const float* shared_down,
    const float* expert_gate_up_stacked, const float* expert_down_stacked,
    uint32_t num_experts, uint32_t intermediate,
    const MoERoutingConfig& routing_cfg,
    float* output)
{
    // Step 1: Router logits
    float* router_logits = new float[n_tokens * num_experts];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < num_experts; e++) {
            float s = 0;
            for (uint32_t h = 0; h < hidden; h++)
                s += hidden_states[t * hidden + h] * router_weight[e * hidden + h];
            router_logits[t * num_experts + e] = s;
        }
    }
    
    // Step 2: Routing
    MoEGateOutput gate_out;
    uint32_t capacity = routing_cfg.drop_tokens ? 
        (uint32_t)(std::max(routing_cfg.moe_topk, 
            (uint32_t)(routing_cfg.capacity_factor * routing_cfg.moe_topk * n_tokens / num_experts)))
        : routing_cfg.moe_topk * n_tokens / num_experts + 1;
    if (capacity < routing_cfg.min_capacity) capacity = routing_cfg.min_capacity;
    
    gate_out.alloc(n_tokens, num_experts, routing_cfg.moe_topk, capacity, true);
    topkgating_full(router_logits, n_tokens, routing_cfg, gate_out);
    delete[] router_logits;
    
    // Step 3: Shared expert
    float* shared_out = nullptr;
    if (shared_gate_up && shared_down) {
        shared_out = new float[n_tokens * hidden];
        // SwiGLU FFN for shared expert (same as swiglu_ffn_f32 from attention.cpp)
        extern void swiglu_ffn_f32(const float*, const float*, const float*, float*, uint32_t, uint32_t, uint32_t);
        swiglu_ffn_f32(hidden_states, shared_gate_up, shared_down, shared_out, n_tokens, hidden, intermediate);
    }
    
    // Step 4: Dispatch input to experts
    float* dispatched = new float[(size_t)num_experts * capacity * hidden];
    einsum_dispatch(hidden_states, n_tokens, hidden, gate_out.dispatch_mask, 
                    num_experts, capacity, dispatched);
    
    // Step 5: Compute expert outputs
    float* expert_outputs = new float[(size_t)num_experts * capacity * hidden]();
    uint32_t mid = 2 * intermediate;
    
    for (uint32_t e = 0; e < num_experts; e++) {
        const float* gate_up = expert_gate_up_stacked + (size_t)e * mid * hidden;
        const float* down = expert_down_stacked + (size_t)e * hidden * intermediate;
        
        for (uint32_t c = 0; c < capacity; c++) {
            const float* token = dispatched + (e * capacity + c) * hidden;
            float* out_token = expert_outputs + (e * capacity + c) * hidden;
            
            // Check if this slot is used (non-zero input)
            bool has_data = false;
            for (uint32_t h = 0; h < hidden; h++)
                if (token[h] != 0) { has_data = true; break; }
            
            if (!has_data) continue;
            
            // SwiGLU FFN
            float* gate_up_out = new float[mid];
            for (uint32_t j = 0; j < mid; j++) {
                float s = 0;
                for (uint32_t h = 0; h < hidden; h++)
                    s += token[h] * gate_up[j * hidden + h];
                gate_up_out[j] = s;
            }
            
            // Separate gate and up, apply silu
            float* hidden_act = new float[intermediate];
            for (uint32_t j = 0; j < intermediate; j++) {
                float g = gate_up_out[j];
                float u = gate_up_out[intermediate + j];
                float silu_g = g * (1.0f / (1.0f + expf(-g)));
                hidden_act[j] = silu_g * u;
            }
            delete[] gate_up_out;
            
            // Down projection
            for (uint32_t j = 0; j < hidden; j++) {
                float s = 0;
                for (uint32_t k = 0; k < intermediate; k++)
                    s += hidden_act[k] * down[j * intermediate + k];
                out_token[j] = s;
            }
            delete[] hidden_act;
        }
    }
    delete[] dispatched;
    
    // Step 6: Combine expert outputs
    float* combined = new float[n_tokens * hidden];
    einsum_combine(gate_out.combine_weights, expert_outputs, n_tokens, hidden,
                   num_experts, capacity, combined);
    delete[] expert_outputs;
    
    // Step 7: Final output (shared + routed)
    if (shared_out) {
        for (uint32_t i = 0; i < n_tokens * hidden; i++)
            output[i] = shared_out[i] + combined[i];
        delete[] shared_out;
    } else {
        memcpy(output, combined, n_tokens * hidden * sizeof(float));
    }
    delete[] combined;
}
