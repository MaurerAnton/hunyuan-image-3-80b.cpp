#include "hunyuan_model.h"
#include <cmath>
#include <cstring>

// ============================================================
// RMS Normalization (pure C++ reference implementation)
// ============================================================

void rms_norm_f32(
    const float* x,
    const float* weight,
    float* out,
    uint32_t n_tokens,
    uint32_t hidden_size,
    float eps)
{
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float* xi = x + i * hidden_size;
        float* oi = out + i * hidden_size;
        
        // Compute RMS: sqrt(mean(x^2))
        float sum_sq = 0.0f;
        for (uint32_t j = 0; j < hidden_size; j++) {
            sum_sq += xi[j] * xi[j];
        }
        float rms = 1.0f / sqrtf(sum_sq / hidden_size + eps);
        
        // Normalize and scale
        for (uint32_t j = 0; j < hidden_size; j++) {
            oi[j] = xi[j] * rms * weight[j];
        }
    }
}

// ============================================================
// SwiGLU FFN
// ============================================================

void swiglu_ffn_f32(
    const float* x,
    const float* gate_up_weight,
    const float* down_weight,
    float* out,
    uint32_t n,
    uint32_t hidden_size,
    uint32_t intermediate_size)
{
    // gate_up_weight: [2*intermediate, hidden_size]
    // x:             [n, hidden_size]
    // tmp = x @ gate_up_weight^T = [n, 2*intermediate]
    uint32_t mid2 = 2 * intermediate_size;
    float* tmp = new float[n * mid2];
    
    // Linear: tmp[i][j] = sum_k x[i][k] * gate_up_weight[j][k]
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < mid2; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < hidden_size; k++) {
                s += x[i * hidden_size + k] * gate_up_weight[j * hidden_size + k];
            }
            tmp[i * mid2 + j] = s;
        }
    }
    
    // Split into gate and up, apply SwiGLU
    // gate_up = silu(gate) * up
    float* gate_up = new float[n * intermediate_size];
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < intermediate_size; j++) {
            float g = tmp[i * mid2 + j];                    // gate part
            float u = tmp[i * mid2 + intermediate_size + j]; // up part
            // SiLU: x * sigmoid(x)
            float silu_g = g * (1.0f / (1.0f + expf(-g)));
            gate_up[i * intermediate_size + j] = silu_g * u;
        }
    }
    delete[] tmp;
    
    // Down projection: out = gate_up @ down_weight^T
    // down_weight: [hidden_size, intermediate]
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < hidden_size; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < intermediate_size; k++) {
                s += gate_up[i * intermediate_size + k] * down_weight[j * intermediate_size + k];
            }
            out[i * hidden_size + j] = s;
        }
    }
    delete[] gate_up;
}

// ============================================================
// Rotary Position Embedding
// ============================================================

void rope_f32(
    float* q,
    float* k,
    uint32_t seq_len,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta,
    const int* positions)
{
    // Precompute frequencies
    float* freq = new float[head_dim / 2];
    for (uint32_t i = 0; i < head_dim / 2; i++) {
        freq[i] = 1.0f / powf(theta, (2.0f * i) / head_dim);
    }
    
    // Apply to Q: [seq_len, n_heads, head_dim]
    for (uint32_t s = 0; s < seq_len; s++) {
        float pos = positions ? (float)positions[s] : (float)s;
        for (uint32_t h = 0; h < n_heads; h++) {
            float* qh = q + (s * n_heads + h) * head_dim;
            float* tmp = new float[head_dim];
            for (uint32_t d = 0; d < head_dim / 2; d++) {
                float cos_a = cosf(pos * freq[d]);
                float sin_a = sinf(pos * freq[d]);
                float q0 = qh[2 * d];
                float q1 = qh[2 * d + 1];
                tmp[2 * d]     = q0 * cos_a - q1 * sin_a;
                tmp[2 * d + 1] = q0 * sin_a + q1 * cos_a;
            }
            memcpy(qh, tmp, head_dim * sizeof(float));
            delete[] tmp;
        }
    }
    
    // Apply to K: [seq_len, n_kv_heads, head_dim]
    for (uint32_t s = 0; s < seq_len; s++) {
        float pos = positions ? (float)positions[s] : (float)s;
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            float* kh = k + (s * n_kv_heads + h) * head_dim;
            float* tmp = new float[head_dim];
            for (uint32_t d = 0; d < head_dim / 2; d++) {
                float cos_a = cosf(pos * freq[d]);
                float sin_a = sinf(pos * freq[d]);
                float k0 = kh[2 * d];
                float k1 = kh[2 * d + 1];
                tmp[2 * d]     = k0 * cos_a - k1 * sin_a;
                tmp[2 * d + 1] = k0 * sin_a + k1 * cos_a;
            }
            memcpy(kh, tmp, head_dim * sizeof(float));
            delete[] tmp;
        }
    }
    
    delete[] freq;
}

// ============================================================
// Helpers
// ============================================================

float tensor_rms(const float* x, uint32_t n) {
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum_sq += (double)x[i] * x[i];
    }
    return (float)sqrt(sum_sq / n);
}

float tensor_correlation(const float* a, const float* b, uint32_t n) {
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0, sum_a2 = 0.0, sum_b2 = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum_a += a[i];
        sum_b += b[i];
        sum_ab += (double)a[i] * b[i];
        sum_a2 += (double)a[i] * a[i];
        sum_b2 += (double)b[i] * b[i];
    }
    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double cov = sum_ab - n * mean_a * mean_b;
    double var_a = sum_a2 - n * mean_a * mean_a;
    double var_b = sum_b2 - n * mean_b * mean_b;
    if (var_a < 1e-30 || var_b < 1e-30) return 0.0f;
    return (float)(cov / sqrt(var_a * var_b));
}

void print_tensor_stats(const char* label, const float* x, uint32_t n) {
    float rms = tensor_rms(x, n);
    // Find min/max
    float vmin = x[0], vmax = x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] < vmin) vmin = x[i];
        if (x[i] > vmax) vmax = x[i];
    }
    printf("%-20s RMS=%.6f  min=%.4f  max=%.4f\n", label, rms, vmin, vmax);
}
