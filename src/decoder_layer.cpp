#include "hunyuan_model.h"
#include <cstring>
#include <cstdlib>

// ============================================================
// Decoder Layer Forward
// ============================================================

void decoder_layer_forward_cpu(
    const float* hidden_states,
    const DecoderLayerWeights& w,
    uint32_t n_tokens,
    const HunyuanConfig& cfg,
    const int* positions,
    float* output)
{
    uint32_t H = cfg.hidden_size;
    uint32_t total = n_tokens * H;
    
    // --- input_layernorm ---
    float* normed = new float[total];
    rms_norm_f32(hidden_states, w.input_ln_weight.data(), normed, n_tokens, H, cfg.rms_norm_eps);
    
    // --- self-attention ---
    float* attn_out = new float[total];
    attention_forward_cpu(normed, w.attn, n_tokens, cfg, positions, attn_out, nullptr, nullptr, nullptr);
    
    // --- residual 1 ---
    float* h = new float[total];
    for (uint32_t i = 0; i < total; i++) h[i] = hidden_states[i] + attn_out[i];
    delete[] attn_out;
    
    // --- post_attention_layernorm ---
    rms_norm_f32(h, w.post_attn_ln_weight.data(), normed, n_tokens, H, cfg.rms_norm_eps);
    
    // --- MoE ---
    float* moe_out = new float[total];
    moe_forward_cpu(normed, w.moe, n_tokens, cfg, moe_out, nullptr, nullptr, nullptr);
    delete[] normed;
    
    // --- residual 2 ---
    for (uint32_t i = 0; i < total; i++) output[i] = h[i] + moe_out[i];
    
    delete[] h;
    delete[] moe_out;
}
