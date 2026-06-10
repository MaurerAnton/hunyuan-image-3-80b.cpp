#include "hunyuan_vit.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

// ============================================================
// LayerNorm
// ============================================================

void layer_norm_f32(
    const float* x, uint32_t n, uint32_t dim,
    const float* weight, const float* bias, float eps,
    float* out)
{
    for (uint32_t i = 0; i < n; i++) {
        const float* xi = x + i * dim;
        float* oi = out + i * dim;
        
        // Mean
        float mean = 0;
        for (uint32_t d = 0; d < dim; d++) mean += xi[d];
        mean /= dim;
        
        // Variance
        float var = 0;
        for (uint32_t d = 0; d < dim; d++) { float diff = xi[d] - mean; var += diff * diff; }
        var /= dim;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(var + eps);
        for (uint32_t d = 0; d < dim; d++)
            oi[d] = (xi[d] - mean) * inv_std * weight[d] + bias[d];
    }
}

// ============================================================
// Utility: Linear + matmul
// ============================================================

static void linear_f32(
    const float* x, uint32_t n, uint32_t in_dim,
    const float* weight, const float* bias, uint32_t out_dim,
    float* out)
{
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < out_dim; j++) {
            float s = bias ? bias[j] : 0;
            for (uint32_t k = 0; k < in_dim; k++)
                s += x[i * in_dim + k] * weight[j * in_dim + k];
            out[i * out_dim + j] = s;
        }
    }
}

// ============================================================
// ViT Self-Attention (single head version, simplified)
// ============================================================

static void vit_attention_f32(
    const float* x, uint32_t n, uint32_t dim, uint32_t n_heads,
    const float* qkv_weight, const float* qkv_bias,
    const float* out_weight, const float* out_bias,
    float* attn_out)
{
    uint32_t head_dim = dim / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // QKV projection: [n, dim] → [n, 3*dim]
    // ViT layout: Q, K, V concatenated along dim
    float* qkv = new float[n * 3 * dim];
    linear_f32(x, n, dim, qkv_weight, qkv_bias, 3 * dim, qkv);
    
    // Reshape: [n, n_heads, 3, head_dim]
    float* qh = new float[n * n_heads * head_dim];
    float* kh = new float[n * n_heads * head_dim];
    float* vh = new float[n * n_heads * head_dim];
    
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                qh[(ni * n_heads + h) * head_dim + d] = qkv[ni * 3 * dim + h * head_dim + d];
                kh[(ni * n_heads + h) * head_dim + d] = qkv[ni * 3 * dim + dim + h * head_dim + d];
                vh[(ni * n_heads + h) * head_dim + d] = qkv[ni * 3 * dim + 2 * dim + h * head_dim + d];
            }
        }
    }
    delete[] qkv;
    
    // Per-head attention, merge heads
    float* out = new float[n * dim]();
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            float* scores = new float[n];
            float max_s = -1e30f;
            for (uint32_t nj = 0; nj < n; nj++) {
                float dot = 0;
                for (uint32_t d = 0; d < head_dim; d++)
                    dot += qh[(ni * n_heads + h) * head_dim + d] * kh[(nj * n_heads + h) * head_dim + d];
                scores[nj] = dot * scale;
                if (scores[nj] > max_s) max_s = scores[nj];
            }
            float sum_e = 0;
            for (uint32_t nj = 0; nj < n; nj++) {
                scores[nj] = expf(scores[nj] - max_s);
                sum_e += scores[nj];
            }
            for (uint32_t nj = 0; nj < n; nj++) scores[nj] /= sum_e;
            
            for (uint32_t d = 0; d < head_dim; d++) {
                float s = 0;
                for (uint32_t nj = 0; nj < n; nj++)
                    s += scores[nj] * vh[(nj * n_heads + h) * head_dim + d];
                out[ni * dim + h * head_dim + d] = s;
            }
            delete[] scores;
        }
    }
    delete[] qh; delete[] kh; delete[] vh;
    
    // Output projection: [n, dim] → [n, dim]
    linear_f32(out, n, dim, out_weight, out_bias, dim, attn_out);
    delete[] out;
}

// ============================================================
// ViT Forward
// ============================================================

void vit_forward_f32(
    const float* image, uint32_t H, uint32_t W,
    const ViTWeights& w,
    float* features)
{
    uint32_t dim = w.hidden_size;
    uint32_t patch = w.patch_size;
    uint32_t n_patches_h = H / patch;
    uint32_t n_patches_w = W / patch;
    uint32_t n_patches = n_patches_h * n_patches_w;
    
    // Patch embedding: Conv2d(3→dim, patch×patch, stride=patch)
    float* x = new float[n_patches * dim]();
    for (uint32_t ph = 0; ph < n_patches_h; ph++) {
        for (uint32_t pw = 0; pw < n_patches_w; pw++) {
            uint32_t pi = ph * n_patches_w + pw;
            for (uint32_t oc = 0; oc < dim; oc++) {
                float sum = w.patch_embed_bias.empty() ? 0 : w.patch_embed_bias[oc];
                for (uint32_t ic = 0; ic < 3; ic++) {
                    for (uint32_t kh = 0; kh < patch; kh++) {
                        for (uint32_t kw = 0; kw < patch; kw++) {
                            int ih = ph * patch + kh;
                            int iw = pw * patch + kw;
                            if (ih < (int)H && iw < (int)W) {
                                sum += image[(ic * H + ih) * W + iw] * 
                                    w.patch_embed_weight[((oc * 3 + ic) * patch + kh) * patch + kw];
                            }
                        }
                    }
                }
                x[pi * dim + oc] = sum;
            }
        }
    }
    
    // Add position embedding
    if (!w.pos_embed.empty()) {
        for (uint32_t i = 0; i < n_patches * dim; i++)
            x[i] += w.pos_embed[i];
    }
    
    // Pre-norm
    if (!w.pre_norm_weight.empty()) {
        float* normed = new float[n_patches * dim];
        layer_norm_f32(x, n_patches, dim, w.pre_norm_weight.data(), w.pre_norm_bias.data(), 1e-6f, normed);
        memcpy(x, normed, n_patches * dim * sizeof(float));
        delete[] normed;
    }
    
    // Transformer layers
    for (uint32_t l = 0; l < w.num_layers && l < w.layers.size(); l++) {
        auto& layer = w.layers[l];
        
        // Self-attention with pre-norm
        float* norm1 = new float[n_patches * dim];
        layer_norm_f32(x, n_patches, dim, layer.norm1_weight.data(), layer.norm1_bias.data(), 1e-6f, norm1);
        
        float* attn_out = new float[n_patches * dim];
        vit_attention_f32(norm1, n_patches, dim, w.num_heads,
                          layer.qkv_weight.data(), layer.qkv_bias.data(),
                          layer.out_weight.data(), layer.out_bias.data(), attn_out);
        delete[] norm1;
        
        // Residual
        for (uint32_t i = 0; i < n_patches * dim; i++) x[i] += attn_out[i];
        delete[] attn_out;
        
        // MLP with pre-norm
        float* norm2 = new float[n_patches * dim];
        layer_norm_f32(x, n_patches, dim, layer.norm2_weight.data(), layer.norm2_bias.data(), 1e-6f, norm2);
        
        float* mlp_hidden = new float[n_patches * w.intermediate_size];
        linear_f32(norm2, n_patches, dim, layer.mlp_fc1_weight.data(), layer.mlp_fc1_bias.data(),
                   w.intermediate_size, mlp_hidden);
        for (uint32_t i = 0; i < n_patches * w.intermediate_size; i++)
            mlp_hidden[i] = gelu_f32(mlp_hidden[i]);
        
        float* mlp_out = new float[n_patches * dim];
        linear_f32(mlp_hidden, n_patches, w.intermediate_size, layer.mlp_fc2_weight.data(), layer.mlp_fc2_bias.data(),
                   dim, mlp_out);
        delete[] norm2; delete[] mlp_hidden;
        
        // Residual
        for (uint32_t i = 0; i < n_patches * dim; i++) x[i] += mlp_out[i];
        delete[] mlp_out;
    }
    
    memcpy(features, x, n_patches * dim * sizeof(float));
    delete[] x;
}

// ============================================================
// Vision Aligner Forward
// ============================================================

void aligner_forward_f32(
    const float* vit_features, uint32_t N,
    const VisionAlignerWeights& w,
    float* output)
{
    uint32_t in_dim = w.input_dim;
    uint32_t out_dim = w.n_embed;
    
    // Layer 0: Linear(in → out) + GELU
    float* h = new float[N * out_dim];
    linear_f32(vit_features, N, in_dim, w.layer0_weight.data(), w.layer0_bias.data(), out_dim, h);
    for (uint32_t i = 0; i < N * out_dim; i++) h[i] = gelu_f32(h[i]);
    
    // Layer 2: Linear(out → out)
    linear_f32(h, N, out_dim, w.layer2_weight.data(), w.layer2_bias.data(), out_dim, output);
    delete[] h;
}
