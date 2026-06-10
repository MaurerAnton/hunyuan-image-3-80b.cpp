#pragma once

#include "hunyuan_model.h"
#include "gguf.h"
#include <cstdio>
#include <string>
#include <vector>
#include <map>

// ============================================================
// GGUF-based weight loader for Hunyuan Image 3
// 
// Supports:
// - Per-layer GGUF files (hunyuan_layer_N.gguf)
// - Global weights file (hunyuan_global.gguf)
// - Auto-dequantization (Q8_0, Q4_0 → F32)
// - Memory-efficient: only loads tensors for current layer
// ============================================================

class HunyuanGGUFLoader {
public:
    HunyuanGGUFLoader() = default;
    ~HunyuanGGUFLoader();
    
    // Open the global weights GGUF file
    bool open_global(const std::string& path);
    
    // Open a per-layer GGUF file (closes previous layer file)
    bool open_layer(const std::string& path);
    
    // Close all files
    void close();
    
    // Load a tensor by name (reads from currently open file)
    bool load_tensor(const std::string& name, std::vector<float>& dst);
    
    // Load all tensors for a layer into DecoderLayerWeights
    bool load_layer_weights(int layer_idx, DecoderLayerWeights& w);
    
    // Check if a tensor exists in currently open file
    bool has_tensor(const std::string& name) const;
    
    // Access tensors loaded from global file
    const std::vector<float>& embedding() const { return global_tensors_.at("token_embd.weight"); }
    const std::vector<float>& final_norm() const { return global_tensors_.at("output_norm.weight"); }
    const std::vector<float>& lm_head() const { return global_tensors_.at("output.weight"); }
    
    // Metadata
    uint32_t get_metadata_u32(const std::string& key, uint32_t default_val = 0) const;
    float    get_metadata_f32(const std::string& key, float default_val = 0.0f) const;
    
private:
    struct OpenFile {
        std::string path;
        FILE* fp = nullptr;
        gguf_context* ctx = nullptr;
        size_t data_offset = 0;
    };
    
    OpenFile current_file_;
    std::map<std::string, std::vector<float>> global_tensors_;
    
    // Helper: read and dequantize tensor data from file
    bool read_tensor_data(const std::string& name, ggml_type type,
                          size_t offset, size_t nbytes, 
                          std::vector<float>& dst);
    
    // Dequantize helpers
    static void dequantize_q8_0(const void* src, float* dst, size_t n);
    static void dequantize_q4_0(const void* src, float* dst, size_t n);
};
