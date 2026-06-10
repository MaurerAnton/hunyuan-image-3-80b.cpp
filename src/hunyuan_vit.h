#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

// ============================================================
// ViT (Vision Transformer) — SigLIP2 SO400M
//
// Encodes conditional images for img2img/inpainting.
// Architecture: 27 layers, hidden=1152, 16 heads, patch=16
// ============================================================

struct ViTWeights {
    uint32_t hidden_size = 1152;
    uint32_t num_layers = 27;
    uint32_t num_heads = 16;
    uint32_t intermediate_size = 4304;
    uint32_t patch_size = 16;
    uint32_t image_size = 384;   // default input size
    uint32_t num_patches = 256;  // (384/16)^2
    
    // Patch embedding: Conv2d(3→1152, 16×16, stride=16)
    std::vector<float> patch_embed_weight;  // [1152, 3, 16, 16]
    std::vector<float> patch_embed_bias;    // [1152]
    
    // Position embedding: [1, num_patches, hidden_size]
    std::vector<float> pos_embed;
    
    // CLS token (optional, not used in Hunyuan's SigLIP)
    bool use_cls = false;
    
    // Pre-norm
    std::vector<float> pre_norm_weight;  // [hidden_size]
    std::vector<float> pre_norm_bias;    // [hidden_size]
    
    // Transformer layers
    struct Layer {
        // Self-attention
        std::vector<float> qkv_weight;    // [3*hidden, hidden]
        std::vector<float> qkv_bias;      // [3*hidden]
        std::vector<float> out_weight;    // [hidden, hidden]
        std::vector<float> out_bias;      // [hidden]
        
        // Layer norms
        std::vector<float> norm1_weight, norm1_bias;
        std::vector<float> norm2_weight, norm2_bias;
        
        // MLP
        std::vector<float> mlp_fc1_weight; // [intermediate, hidden]
        std::vector<float> mlp_fc1_bias;
        std::vector<float> mlp_fc2_weight; // [hidden, intermediate]
        std::vector<float> mlp_fc2_bias;
    };
    std::vector<Layer> layers;
};

// ============================================================
// Vision Aligner (LightProjector)
//
// Projects ViT features [1152] → transformer space [4096]
// 2-layer MLP with GELU
// ============================================================

struct VisionAlignerWeights {
    uint32_t input_dim = 1152;
    uint32_t n_embed = 4096;
    uint32_t depth = 2;   // number of MLP layers
    
    // layers.0: Linear(input_dim → n_embed)
    std::vector<float> layer0_weight;  // [n_embed, input_dim]
    std::vector<float> layer0_bias;    // [n_embed]
    
    // layers.2: Linear(n_embed → n_embed)
    std::vector<float> layer2_weight;  // [n_embed, n_embed]
    std::vector<float> layer2_bias;    // [n_embed]
};

// ============================================================
// ViT forward
// ============================================================

void vit_forward_f32(
    const float* image,       // [1, 3, H, W] — RGB, normalized to [-1, 1]
    uint32_t H, uint32_t W,
    const ViTWeights& w,
    float* features);         // [num_patches, hidden_size]

// ============================================================
// Vision Aligner forward
// ============================================================

void aligner_forward_f32(
    const float* vit_features,  // [N, input_dim]
    uint32_t N,
    const VisionAlignerWeights& w,
    float* output);             // [N, n_embed]

// ============================================================
// LayerNorm (not RMSNorm — ViT uses standard LayerNorm)
// ============================================================

void layer_norm_f32(
    const float* x, uint32_t n, uint32_t dim,
    const float* weight, const float* bias, float eps,
    float* out);

// ============================================================
// GELU activation (tanh approximation)
// ============================================================

inline float gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
