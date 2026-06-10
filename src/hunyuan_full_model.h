#pragma once

#include "hunyuan_model.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include <map>

// ============================================================
// Weight provider: loads layers on-demand from safetensors shards
// This enables running the 80B model on 128GB RAM by loading
// only one layer at a time.
// ============================================================

class HunyuanWeightLoader {
public:
    HunyuanWeightLoader(const std::string& weights_dir, const std::string& index_path);
    
    // Load embeddings + final norm + lm_head (always needed)
    bool load_global_weights();
    
    // Load one decoder layer's weights (frees previous layer)
    bool load_layer(int layer_idx, DecoderLayerWeights& w);
    
    // Release layer weights to free memory
    void release_layer(DecoderLayerWeights& w);
    
    // Access global weights
    const std::vector<float>& embedding() const { return wte_weight; }
    const std::vector<float>& final_norm() const { return ln_f_weight; }
    const std::vector<float>& lm_head() const { return lm_head_weight; }
    
    // Shard cache: keep open file handles for frequently-accessed shards
    std::string get_shard_for_tensor(const std::string& tensor_name) const;
    
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t num_layers = 0;
    
private:
    std::string weights_dir;
    std::vector<float> wte_weight;    // [vocab_size, hidden_size] — embedding lookup
    std::vector<float> ln_f_weight;   // [hidden_size] — final RMSNorm
    std::vector<float> lm_head_weight; // [vocab_size, hidden_size] — output projection
    
    // Index: tensor_name → shard_file
    std::map<std::string, std::string> weight_map;
};

// ============================================================
// Full Hunyuan Image 3 Model (transformer backbone only)
// ============================================================

class HunyuanFullModel {
public:
    HunyuanFullModel(const HunyuanConfig& cfg);
    ~HunyuanFullModel();
    
    // Initialize with weight loader
    bool init(HunyuanWeightLoader* loader);
    
    // Forward pass: input_ids → logits
    // Uses layer-by-layer streaming: loads, computes, frees each layer
    void forward(
        const int* input_ids,        // [n_tokens]
        uint32_t n_tokens,
        float* logits,               // [n_tokens, vocab_size]
        const int* positions = nullptr);
    
    // Forward pass returning hidden states (for image generation pipeline)
    void forward_hidden(
        const float* inputs_embeds,   // [n_tokens, hidden_size]
        uint32_t n_tokens,
        float* hidden_states,        // [n_tokens, hidden_size]
        const int* positions = nullptr);
    
    // Get hidden states from a specific layer (for debugging)
    void forward_debug(
        const float* inputs_embeds,
        uint32_t n_tokens,
        int target_layer,            // -1 = all layers, 0..31 = specific
        std::vector<float>& hidden_out,
        const int* positions = nullptr);
    
    const HunyuanConfig& config() const { return cfg; }
    
private:
    HunyuanConfig cfg;
    HunyuanWeightLoader* loader = nullptr;
    
    // Scratch buffers (reused across layers)
    float* hidden_buf = nullptr;     // [n_tokens, hidden_size]
    float* layer_out = nullptr;      // [n_tokens, hidden_size]
    float* norm_buf = nullptr;       // [n_tokens, hidden_size]
    uint32_t buf_n_tokens = 0;
    
    void ensure_buffers(uint32_t n_tokens);
};

// ============================================================
// Embedding lookup
// ============================================================
void embedding_lookup_f32(
    const int* input_ids,           // [n_tokens]
    const float* embedding_table,   // [vocab_size, hidden_size]
    float* out,                     // [n_tokens, hidden_size]
    uint32_t n_tokens,
    uint32_t hidden_size);

// ============================================================
// LM Head (output projection to vocabulary)
// ============================================================
void lm_head_f32(
    const float* hidden_states,     // [n_tokens, hidden_size]
    const float* lm_head_weight,    // [vocab_size, hidden_size]
    float* logits,                  // [n_tokens, vocab_size]
    uint32_t n_tokens,
    uint32_t hidden_size,
    uint32_t vocab_size);
