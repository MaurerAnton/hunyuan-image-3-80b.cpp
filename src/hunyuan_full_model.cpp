#include "hunyuan_full_model.h"
#include "hunyuan_model.h"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

// ============================================================
// HunyuanWeightLoader
// ============================================================

HunyuanWeightLoader::HunyuanWeightLoader(const std::string& weights_dir, const std::string& index_path)
    : weights_dir(weights_dir)
{
    // Parse index to build weight_map
    std::ifstream f(index_path);
    if (!f) {
        fprintf(stderr, "Cannot open index: %s\n", index_path.c_str());
        return;
    }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    // Minimal JSON parsing for weight_map
    size_t pos = json.find("\"weight_map\"");
    if (pos == std::string::npos) return;
    pos = json.find('{', pos);
    if (pos == std::string::npos) return;
    size_t end = json.find('}', pos);
    std::string map_str = json.substr(pos + 1, end - pos - 1);
    
    // Parse "key": "value" pairs
    size_t i = 0;
    while (i < map_str.size()) {
        while (i < map_str.size() && (map_str[i] == ' ' || map_str[i] == '\n' || map_str[i] == ',')) i++;
        if (i >= map_str.size()) break;
        if (map_str[i] == '"') {
            size_t keystart = i + 1;
            size_t keyend = map_str.find('"', keystart);
            std::string key = map_str.substr(keystart, keyend - keystart);
            i = map_str.find('"', keyend + 1);
            size_t valstart = i + 1;
            size_t valend = map_str.find('"', valstart);
            std::string val = map_str.substr(valstart, valend - valstart);
            weight_map[key] = val;
            i = valend + 1;
        } else {
            i++;
        }
    }
    printf("WeightLoader: loaded %zu tensor mappings\n", weight_map.size());
}

std::string HunyuanWeightLoader::get_shard_for_tensor(const std::string& tensor_name) const {
    auto it = weight_map.find(tensor_name);
    if (it != weight_map.end()) return weights_dir + "/" + it->second;
    return "";
}

bool HunyuanWeightLoader::load_global_weights() {
    // Load from safetensors
    // embedding: model.wte.weight
    // final norm: model.ln_f.weight
    // lm_head: lm_head.weight
    
    auto load_one = [this](const std::string& name, std::vector<float>& dst) -> bool {
        std::string shard = get_shard_for_tensor(name);
        if (shard.empty()) {
            fprintf(stderr, "Tensor %s not in index\n", name.c_str());
            return false;
        }
        // Use the SafetensorsFile from safetensors_loader.cpp
        extern bool safetensors_load_tensor(const std::string& path, const std::string& name, std::vector<float>& dst);
        return safetensors_load_tensor(shard, name, dst);
    };
    
    if (!load_one("model.wte.weight", wte_weight)) return false;
    if (!load_one("model.ln_f.weight", ln_f_weight)) return false;
    if (!load_one("lm_head.weight", lm_head_weight)) return false;
    
    // Determine dimensions
    if (!wte_weight.empty()) {
        // wte is [vocab_size, hidden_size]
        // We need to know the shape. The tensor was loaded but we didn't track original shape.
        // For now, assume from config.
        vocab_size = 133120;
        hidden_size = 4096;
    }
    
    printf("Global weights loaded: wte=%zu ln_f=%zu lm_head=%zu\n",
           wte_weight.size(), ln_f_weight.size(), lm_head_weight.size());
    return true;
}

bool HunyuanWeightLoader::load_layer(int layer_idx, DecoderLayerWeights& w) {
    // Build comma-separated list of shards needed for this layer
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", layer_idx);
    std::string pfx(prefix);
    
    // Find all unique shards
    std::set<std::string> shard_set;
    for (auto& [name, shard] : weight_map) {
        if (name.find(pfx) == 0) {
            shard_set.insert(shard);
        }
    }
    
    std::string shard_list;
    for (auto& s : shard_set) {
        if (!shard_list.empty()) shard_list += " ";
        shard_list += weights_dir + "/" + s;
    }
    
    return w.load(layer_idx, shard_list);
}

void HunyuanWeightLoader::release_layer(DecoderLayerWeights& w) {
    // Clear all vectors to free memory
    w.input_ln_weight.clear(); w.input_ln_weight.shrink_to_fit();
    w.post_attn_ln_weight.clear(); w.post_attn_ln_weight.shrink_to_fit();
    w.attn.qkv_weight.clear(); w.attn.qkv_weight.shrink_to_fit();
    w.attn.o_weight.clear(); w.attn.o_weight.shrink_to_fit();
    w.attn.q_norm_weight.clear(); w.attn.q_norm_weight.shrink_to_fit();
    w.attn.k_norm_weight.clear(); w.attn.k_norm_weight.shrink_to_fit();
    w.moe.shared_gate_up.clear(); w.moe.shared_gate_up.shrink_to_fit();
    w.moe.shared_down.clear(); w.moe.shared_down.shrink_to_fit();
    w.moe.router_weight.clear(); w.moe.router_weight.shrink_to_fit();
    for (auto& e : w.moe.experts) {
        e.gate_up.clear(); e.gate_up.shrink_to_fit();
        e.down.clear(); e.down.shrink_to_fit();
    }
    w.moe.experts.clear();
}

// ============================================================
// HunyuanFullModel
// ============================================================

HunyuanFullModel::HunyuanFullModel(const HunyuanConfig& cfg) : cfg(cfg) {}

HunyuanFullModel::~HunyuanFullModel() {
    delete[] hidden_buf;
    delete[] layer_out;
    delete[] norm_buf;
}

bool HunyuanFullModel::init(HunyuanWeightLoader* ldr) {
    loader = ldr;
    if (loader) {
        cfg.hidden_size = loader->hidden_size;
        cfg.vocab_size = loader->vocab_size;
        cfg.num_hidden_layers = loader->num_layers;
    }
    cfg.compute_derived();
    return true;
}

void HunyuanFullModel::ensure_buffers(uint32_t n_tokens) {
    if (n_tokens != buf_n_tokens) {
        delete[] hidden_buf;
        delete[] layer_out;
        delete[] norm_buf;
        buf_n_tokens = n_tokens;
        size_t layer_size = n_tokens * cfg.hidden_size;
        hidden_buf = new float[layer_size];
        layer_out = new float[layer_size];
        norm_buf = new float[layer_size];
    }
}

void HunyuanFullModel::forward(
    const int* input_ids,
    uint32_t n_tokens,
    float* logits,
    const int* positions)
{
    ensure_buffers(n_tokens);
    
    // Embedding lookup
    embedding_lookup_f32(input_ids, loader->embedding().data(), 
                         hidden_buf, n_tokens, cfg.hidden_size);
    
    // Generate positions if not provided
    std::vector<int> default_pos;
    if (!positions) {
        default_pos.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) default_pos[i] = i;
        positions = default_pos.data();
    }
    
    // Layer-by-layer: load, compute, free
    DecoderLayerWeights w;
    for (uint32_t layer = 0; layer < cfg.num_hidden_layers; layer++) {
        printf("  Layer %d/%d...\n", layer, cfg.num_hidden_layers);
        
        if (!loader->load_layer(layer, w)) {
            fprintf(stderr, "Failed to load layer %d\n", layer);
            return;
        }
        
        decoder_layer_forward_cpu(hidden_buf, w, n_tokens, cfg, positions, layer_out);
        memcpy(hidden_buf, layer_out, n_tokens * cfg.hidden_size * sizeof(float));
        
        loader->release_layer(w);
    }
    
    // Final norm
    rms_norm_f32(hidden_buf, loader->final_norm().data(), norm_buf, 
                 n_tokens, cfg.hidden_size, cfg.rms_norm_eps);
    
    // LM head
    lm_head_f32(norm_buf, loader->lm_head().data(), logits,
                n_tokens, cfg.hidden_size, cfg.vocab_size);
}

void HunyuanFullModel::forward_hidden(
    const float* inputs_embeds,
    uint32_t n_tokens,
    float* hidden_states,
    const int* positions)
{
    ensure_buffers(n_tokens);
    memcpy(hidden_buf, inputs_embeds, n_tokens * cfg.hidden_size * sizeof(float));
    
    std::vector<int> default_pos;
    if (!positions) {
        default_pos.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) default_pos[i] = i;
        positions = default_pos.data();
    }
    
    DecoderLayerWeights w;
    for (uint32_t layer = 0; layer < cfg.num_hidden_layers; layer++) {
        if (!loader->load_layer(layer, w)) return;
        decoder_layer_forward_cpu(hidden_buf, w, n_tokens, cfg, positions, layer_out);
        memcpy(hidden_buf, layer_out, n_tokens * cfg.hidden_size * sizeof(float));
        loader->release_layer(w);
    }
    
    memcpy(hidden_states, hidden_buf, n_tokens * cfg.hidden_size * sizeof(float));
}

void HunyuanFullModel::forward_debug(
    const float* inputs_embeds,
    uint32_t n_tokens,
    int target_layer,
    std::vector<float>& hidden_out,
    const int* positions)
{
    ensure_buffers(n_tokens);
    memcpy(hidden_buf, inputs_embeds, n_tokens * cfg.hidden_size * sizeof(float));
    
    std::vector<int> default_pos;
    if (!positions) {
        default_pos.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) default_pos[i] = i;
        positions = default_pos.data();
    }
    
    DecoderLayerWeights w;
    for (uint32_t layer = 0; layer < cfg.num_hidden_layers; layer++) {
        if (!loader->load_layer(layer, w)) return;
        decoder_layer_forward_cpu(hidden_buf, w, n_tokens, cfg, positions, layer_out);
        memcpy(hidden_buf, layer_out, n_tokens * cfg.hidden_size * sizeof(float));
        loader->release_layer(w);
        
        if (target_layer >= 0 && (int)layer == target_layer) break;
    }
    
    hidden_out.resize(n_tokens * cfg.hidden_size);
    memcpy(hidden_out.data(), hidden_buf, n_tokens * cfg.hidden_size * sizeof(float));
}

// ============================================================
// Embedding Lookup
// ============================================================

void embedding_lookup_f32(
    const int* input_ids,
    const float* embedding_table,
    float* out,
    uint32_t n_tokens,
    uint32_t hidden_size)
{
    for (uint32_t i = 0; i < n_tokens; i++) {
        int token_id = input_ids[i];
        const float* row = embedding_table + token_id * hidden_size;
        memcpy(out + i * hidden_size, row, hidden_size * sizeof(float));
    }
}

// ============================================================
// LM Head
// ============================================================

void lm_head_f32(
    const float* hidden_states,
    const float* lm_head_weight,
    float* logits,
    uint32_t n_tokens,
    uint32_t hidden_size,
    uint32_t vocab_size)
{
    // lm_head_weight: [vocab_size, hidden_size]
    // hidden_states:  [n_tokens, hidden_size]
    // logits = hidden_states @ lm_head_weight^T
    for (uint32_t i = 0; i < n_tokens; i++) {
        for (uint32_t v = 0; v < vocab_size; v++) {
            float s = 0.0f;
            for (uint32_t h = 0; h < hidden_size; h++) {
                s += hidden_states[i * hidden_size + h] * lm_head_weight[v * hidden_size + h];
            }
            logits[i * vocab_size + v] = s;
        }
    }
}
