#include "hunyuan_image_gen.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// ============================================================
// 2D Rotary Position Embedding
// ============================================================

void build_2d_rope(
    uint32_t seq_len,
    uint32_t n_elem,
    uint32_t text_len,
    uint32_t img_h,
    uint32_t img_w,
    float base,
    float* cos_out,
    float* sin_out)
{
    // Compute theta frequencies
    // theta = 1.0 / (base^(arange(0, n_elem, 2) / n_elem))
    // reshaped to [1, n_elem/4, 2] for separate x/y components
    uint32_t half_d = n_elem / 2;
    uint32_t quarter_d = n_elem / 4;
    
    float* theta = new float[half_d];
    for (uint32_t i = 0; i < half_d; i++) {
        theta[i] = 1.0f / powf(base, (2.0f * i) / (float)n_elem);
    }
    
    // For each position, compute (y_pos, x_pos)
    for (uint32_t p = 0; p < seq_len; p++) {
        float y_pos, x_pos;
        
        if (p < text_len) {
            // Text tokens: sequential 1D position (same for x and y)
            y_pos = (float)p;
            x_pos = (float)p;
        } else {
            // Image tokens: 2D grid position
            uint32_t img_idx = p - text_len;
            uint32_t row = img_idx / img_w;
            uint32_t col = img_idx % img_w;
            
            // beta_y = text_len + (img_h*img_w - img_h)/2
            // beta_x = text_len + (img_h*img_w - img_w)/2
            float beta_y = text_len + (img_h * img_w - img_h) / 2.0f;
            float beta_x = text_len + (img_h * img_w - img_w) / 2.0f;
            
            y_pos = beta_y + row;
            x_pos = beta_x + col;
        }
        
        // Compute cos/sin for this position
        for (uint32_t d = 0; d < half_d; d++) {
            // Alternate between y and x for each pair of dims
            // dims 0,1 use y; dims 2,3 use x; dims 4,5 use y; etc.
            float pos = (d / 2) % 2 == 0 ? y_pos : x_pos;
            float angle = pos * theta[d];
            cos_out[p * n_elem + 2*d]     = cosf(angle);
            cos_out[p * n_elem + 2*d + 1] = cosf(angle);
            sin_out[p * n_elem + 2*d]     = sinf(angle);
            sin_out[p * n_elem + 2*d + 1] = sinf(angle);
        }
    }
    
    delete[] theta;
}

void apply_rope_2d_f32(
    float* q,
    float* k,
    const float* cos,
    const float* sin,
    uint32_t n_tokens,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    // Apply to Q: [n_tokens, n_heads, head_dim]
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            float* qh = q + (t * n_heads + h) * head_dim;
            const float* ct = cos + t * head_dim;
            const float* st = sin + t * head_dim;
            float* tmp = new float[head_dim];
            for (uint32_t d = 0; d < head_dim / 2; d++) {
                float q0 = qh[2*d], q1 = qh[2*d+1];
                float c = ct[2*d], s = st[2*d];
                tmp[2*d]   = q0 * c - q1 * s;
                tmp[2*d+1] = q0 * s + q1 * c;
            }
            memcpy(qh, tmp, head_dim * sizeof(float));
            delete[] tmp;
        }
    }
    
    // Apply to K: [n_tokens, n_kv_heads, head_dim]
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            float* kh = k + (t * n_kv_heads + h) * head_dim;
            const float* ct = cos + t * head_dim;
            const float* st = sin + t * head_dim;
            float* tmp = new float[head_dim];
            for (uint32_t d = 0; d < head_dim / 2; d++) {
                float k0 = kh[2*d], k1 = kh[2*d+1];
                float c = ct[2*d], s = st[2*d];
                tmp[2*d]   = k0 * c - k1 * s;
                tmp[2*d+1] = k0 * s + k1 * c;
            }
            memcpy(kh, tmp, head_dim * sizeof(float));
            delete[] tmp;
        }
    }
}

// ============================================================
// Timestep Embedder
// ============================================================

static void sinusoidal_embedding(const float* t, uint32_t batch, 
                                  uint32_t dim, float max_period, float* out) {
    uint32_t half = dim / 2;
    float* freqs = new float[half];
    for (uint32_t i = 0; i < half; i++) {
        freqs[i] = expf(-logf(max_period) * i / half);
    }
    
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t i = 0; i < half; i++) {
            float angle = t[b] * freqs[i];
            out[b * dim + i] = cosf(angle);
            out[b * dim + half + i] = sinf(angle);
        }
    }
    delete[] freqs;
}

void timestep_embedding_f32(
    const float* t,
    uint32_t batch,
    const TimestepEmbedderWeights& w,
    float* out)
{
    // Step 1: sinusoidal embedding [batch, 256]
    float* sino = new float[batch * w.frequency_embedding_size];
    sinusoidal_embedding(t, batch, w.frequency_embedding_size, w.max_period, sino);
    
    // Step 2: Linear(256 → hidden_size) + GELU
    float* hidden = new float[batch * w.hidden_size];
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t j = 0; j < w.hidden_size; j++) {
            float s = w.mlp_0_bias[j];
            for (uint32_t k = 0; k < w.frequency_embedding_size; k++) {
                s += sino[b * w.frequency_embedding_size + k] * w.mlp_0_weight[j * w.frequency_embedding_size + k];
            }
            // GELU approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
            float x = s;
            float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
            hidden[b * w.hidden_size + j] = gelu;
        }
    }
    
    // Step 3: Linear(hidden_size → out_size)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t j = 0; j < w.out_size; j++) {
            float s = w.mlp_2_bias[j];
            for (uint32_t k = 0; k < w.hidden_size; k++) {
                s += hidden[b * w.hidden_size + k] * w.mlp_2_weight[j * w.hidden_size + k];
            }
            out[b * w.out_size + j] = s;
        }
    }
    
    delete[] sino;
    delete[] hidden;
}

// ============================================================
// GroupNorm
// ============================================================

void group_norm_f32(
    const float* x, uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    uint32_t num_groups,
    const float* weight, const float* bias,
    float eps, float* out)
{
    uint32_t ch_per_group = C / num_groups;
    uint32_t spatial = H * W;
    
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t g = 0; g < num_groups; g++) {
            uint32_t ch_start = g * ch_per_group;
            
            // Compute mean and variance
            float sum = 0, sum_sq = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                for (uint32_t s = 0; s < spatial; s++) {
                    float v = x[((b * C + c) * H + s / W) * W + s % W];
                    // Actually: x[b][c][h][w] = x[b*C*H*W + c*H*W + h*W + w]
                }
            }
            
            // Proper indexing: x[b][c][h][w] = x[((b*C + c)*H + h)*W + w]
            float mean = 0, var = 0;
            uint32_t n = ch_per_group * spatial;
            float* vals = new float[n];
            uint32_t idx = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                for (uint32_t hw = 0; hw < spatial; hw++) {
                    float v = x[((b * C + c) * H + (hw / W)) * W + (hw % W)];
                    vals[idx++] = v;
                    mean += v;
                }
            }
            mean /= n;
            for (uint32_t i = 0; i < n; i++) {
                float d = vals[i] - mean;
                var += d * d;
            }
            var = var / n;
            float inv_std = 1.0f / sqrtf(var + eps);
            
            // Normalize, scale, shift
            idx = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                for (uint32_t hw = 0; hw < spatial; hw++) {
                    float normed = (vals[idx++] - mean) * inv_std;
                    out[((b * C + c) * H + (hw / W)) * W + (hw % W)] = normed * weight[c] + bias[c];
                }
            }
            delete[] vals;
        }
    }
}

// ============================================================
// Conv2d (basic implementation)
// ============================================================

void conv2d_f32(
    const float* x, uint32_t B, uint32_t in_ch, uint32_t H, uint32_t W,
    const float* weight, const float* bias,
    uint32_t out_ch, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding,
    float* out)
{
    uint32_t H_out = (H + 2*padding - kH) / stride + 1;
    uint32_t W_out = (W + 2*padding - kW) / stride + 1;
    
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t oc = 0; oc < out_ch; oc++) {
            for (uint32_t oh = 0; oh < H_out; oh++) {
                for (uint32_t ow = 0; ow < W_out; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;
                    for (uint32_t ic = 0; ic < in_ch; ic++) {
                        for (uint32_t kh = 0; kh < kH; kh++) {
                            for (uint32_t kw = 0; kw < kW; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                    float xv = x[((b * in_ch + ic) * H + ih) * W + iw];
                                    float wv = weight[((oc * in_ch + ic) * kH + kh) * kW + kw];
                                    sum += xv * wv;
                                }
                            }
                        }
                    }
                    out[((b * out_ch + oc) * H_out + oh) * W_out + ow] = sum;
                }
            }
        }
    }
}

// ============================================================
// ResBlock
// ============================================================

void resblock_forward_f32(
    const float* x, uint32_t B, uint32_t H, uint32_t W,
    const float* emb,
    const ResBlockWeights& w,
    float* out)
{
    uint32_t in_ch = w.in_channels;
    uint32_t out_ch = w.out_channels;
    
    // Step 1: GroupNorm(in_channels)
    float* normed = new float[B * in_ch * H * W];
    group_norm_f32(x, B, in_ch, H, W, 32, w.norm_weight.data(), w.norm_bias.data(), 1e-5f, normed);
    
    // Step 2: SiLU (element-wise)
    for (uint32_t i = 0; i < B * in_ch * H * W; i++) {
        float v = normed[i];
        normed[i] = v * (1.0f / (1.0f + expf(-v)));
    }
    
    // Step 3: Conv2d(in_ch → out_ch, 3×3, padding=1)
    float* conv_out = new float[B * out_ch * H * W];
    conv2d_f32(normed, B, in_ch, H, W, w.conv_weight.data(), w.conv_bias.data(),
               out_ch, 3, 3, 1, 1, conv_out);
    delete[] normed;
    
    // Step 4: Timestep conditioning — emb → scale, shift
    // emb_layers: Linear(emb_ch → 2*out_ch)
    float* scale = new float[out_ch];
    float* shift = new float[out_ch];
    for (uint32_t j = 0; j < out_ch; j++) {
        scale[j] = w.emb_bias[j] + 1.0f;  // default scale is 1
        shift[j] = w.emb_bias[out_ch + j];
        for (uint32_t k = 0; k < w.emb_channels; k++) {
            scale[j] += emb[k] * w.emb_weight[j * w.emb_channels + k];
            shift[j] += emb[k] * w.emb_weight[(out_ch + j) * w.emb_channels + k];
        }
    }
    
    // Apply scale+shift
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < out_ch; c++) {
            for (uint32_t hw = 0; hw < H * W; hw++) {
                uint32_t i = ((b * out_ch + c) * H + hw / W) * W + hw % W;
                conv_out[i] = conv_out[i] * (1.0f + scale[c]) + shift[c];
            }
        }
    }
    delete[] scale;
    delete[] shift;
    
    // Step 5: Skip connection
    if (in_ch == out_ch) {
        // Identity skip
        for (uint32_t i = 0; i < B * out_ch * H * W; i++) {
            out[i] = conv_out[i] + x[i];
        }
    } else {
        // 1×1 conv skip
        float* skip = new float[B * out_ch * H * W];
        conv2d_f32(x, B, in_ch, H, W, w.shortcut_weight.data(), w.shortcut_bias.data(),
                   out_ch, 1, 1, 1, 0, skip);
        for (uint32_t i = 0; i < B * out_ch * H * W; i++) {
            out[i] = conv_out[i] + skip[i];
        }
        delete[] skip;
    }
    delete[] conv_out;
}

// ============================================================
// Downsample / Upsample
// ============================================================

void downsample_conv_f32(
    const float* x, uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    const float* conv_weight, const float* conv_bias,
    float* out)
{
    // Conv2d with stride=2
    conv2d_f32(x, B, C, H, W, conv_weight, conv_bias, C, 3, 3, 2, 1, out);
}

void upsample_nearest_f32(
    const float* x, uint32_t B, uint32_t C, uint32_t H, uint32_t W,
    float* out)
{
    // Nearest-neighbor 2× upsampling
    uint32_t H2 = H * 2, W2 = W * 2;
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    float v = x[((b * C + c) * H + h) * W + w];
                    for (uint32_t dh = 0; dh < 2; dh++) {
                        for (uint32_t dw = 0; dw < 2; dw++) {
                            out[((b * C + c) * H2 + h*2+dh) * W2 + w*2+dw] = v;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// UNetDown (patch_embed)
// ============================================================

void unet_down_forward_f32(
    const float* x, uint32_t B, uint32_t H, uint32_t W,
    const float* timestep_emb,
    const UNetDownWeights& w,
    float* out)
{
    uint32_t in_ch = w.in_channels;
    uint32_t hid_ch = w.hidden_channels;
    uint32_t out_ch = w.out_channels;
    uint32_t emb_ch = w.emb_channels;
    uint32_t cur_H = H, cur_W = W;
    uint32_t cur_ch = in_ch;
    
    // Project timestep emb to hidden_channels
    // time_emb_proj: Linear(emb_ch → hid_ch)
    float* t_emb_hid = new float[B * hid_ch];
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t j = 0; j < hid_ch; j++) {
            float s = w.time_emb_proj_bias[j];
            for (uint32_t k = 0; k < emb_ch; k++) {
                s += timestep_emb[b * emb_ch + k] * w.time_emb_proj_weight[j * emb_ch + k];
            }
            // SiLU
            float sv = s * (1.0f / (1.0f + expf(-s)));
            t_emb_hid[b * hid_ch + j] = sv;
        }
    }
    
    float* cur = new float[B * cur_ch * cur_H * cur_W];
    memcpy(cur, x, B * cur_ch * cur_H * cur_W * sizeof(float));
    
    // Process blocks
    // For patch_size=1: 6 ResBlocks, no downsampling
    // Block 0-1: in_ch+emb → hid_ch, hid_ch → hid_ch
    // Block 2-3: hid_ch → hid_ch, hid_ch → hid_ch
    // Block 4-5: hid_ch → hid_ch, hid_ch → out_ch
    uint32_t block_idx = 0;
    
    // Block 0: ResBlock(in_ch → hid_ch) with emb concatenated
    {
        // Concatenate emb to input: [B, in_ch+hid_ch, H, W]
        uint32_t cat_ch = in_ch + hid_ch;
        float* cat_in = new float[B * cat_ch * cur_H * cur_W];
        for (uint32_t b = 0; b < B; b++) {
            for (uint32_t c = 0; c < in_ch; c++) {
                memcpy(cat_in + ((b * cat_ch + c) * cur_H) * cur_W,
                       cur + ((b * in_ch + c) * cur_H) * cur_W,
                       cur_H * cur_W * sizeof(float));
            }
            for (uint32_t c = 0; c < hid_ch; c++) {
                for (uint32_t hw = 0; hw < cur_H * cur_W; hw++) {
                    cat_in[((b * cat_ch + in_ch + c) * cur_H + hw / cur_W) * cur_W + hw % cur_W] = 
                        t_emb_hid[b * hid_ch + c];
                }
            }
        }
        
        // ResBlock(cat_ch → hid_ch)
        ResBlockWeights rw = w.blocks[block_idx++];
        rw.in_channels = cat_ch;
        rw.out_channels = hid_ch;
        float* next = new float[B * hid_ch * cur_H * cur_W];
        resblock_forward_f32(cat_in, B, cur_H, cur_W, t_emb_hid, rw, next);
        delete[] cat_in;
        delete[] cur;
        cur = next;
        cur_ch = hid_ch;
    }
    
    // Blocks 1-5: ResBlock(cur_ch → cur_ch) or ResBlock(cur_ch → out_ch)
    for (; block_idx < w.blocks.size(); block_idx++) {
        ResBlockWeights rw = w.blocks[block_idx];
        rw.in_channels = cur_ch;
        
        float* next = new float[B * rw.out_channels * cur_H * cur_W];
        resblock_forward_f32(cur, B, cur_H, cur_W, t_emb_hid, rw, next);
        delete[] cur;
        cur = next;
        cur_ch = rw.out_channels;
    }
    
    // Copy to output
    memcpy(out, cur, B * cur_ch * cur_H * cur_W * sizeof(float));
    delete[] cur;
    delete[] t_emb_hid;
}

// ============================================================
// UNetUp (final_layer)
// ============================================================

void unet_up_forward_f32(
    const float* x, uint32_t B, uint32_t H, uint32_t W,
    const float* timestep_emb,
    const UNetUpWeights& w,
    float* out)
{
    uint32_t in_ch = w.in_channels;
    uint32_t hid_ch = w.hidden_channels;
    uint32_t out_ch = w.out_channels;
    uint32_t emb_ch = w.emb_channels;
    uint32_t cur_H = H, cur_W = W;
    uint32_t cur_ch = in_ch;
    
    // Project timestep emb
    float* t_emb_hid = new float[B * hid_ch];
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t j = 0; j < hid_ch; j++) {
            float s = w.time_emb_proj_bias.empty() ? 0.0f : w.time_emb_proj_bias[j];
            for (uint32_t k = 0; k < emb_ch; k++) {
                if (!w.time_emb_proj_weight.empty())
                    s += timestep_emb[b * emb_ch + k] * w.time_emb_proj_weight[j * emb_ch + k];
            }
            float sv = s * (1.0f / (1.0f + expf(-s)));
            t_emb_hid[b * hid_ch + j] = sv;
        }
    }
    
    float* cur = new float[B * cur_ch * cur_H * cur_W];
    memcpy(cur, x, B * cur_ch * cur_H * cur_W * sizeof(float));
    
    // Process ResBlocks
    for (size_t bi = 0; bi < w.blocks.size(); bi++) {
        ResBlockWeights rw = w.blocks[bi];
        rw.in_channels = cur_ch;
        
        float* next = new float[B * rw.out_channels * cur_H * cur_W];
        resblock_forward_f32(cur, B, cur_H, cur_W, t_emb_hid, rw, next);
        delete[] cur;
        cur = next;
        cur_ch = rw.out_channels;
    }
    
    // Final norm if needed
    if (w.out_norm && !w.out_norm_weight.empty()) {
        float* normed = new float[B * cur_ch * cur_H * cur_W];
        group_norm_f32(cur, B, cur_ch, cur_H, cur_W, 32,
                       w.out_norm_weight.data(), w.out_norm_bias.data(), 1e-5f, normed);
        // SiLU
        for (uint32_t i = 0; i < B * cur_ch * cur_H * cur_W; i++) {
            float v = normed[i];
            normed[i] = v * (1.0f / (1.0f + expf(-v)));
        }
        delete[] cur;
        cur = normed;
    }
    
    memcpy(out, cur, B * cur_ch * cur_H * cur_W * sizeof(float));
    delete[] cur;
    delete[] t_emb_hid;
}

// ============================================================
// KV Cache
// ============================================================

void KVCache::init(uint32_t layers, uint32_t kv_heads, uint32_t hdim, uint32_t max_len) {
    n_layers = layers;
    n_kv_heads = kv_heads;
    head_dim = hdim;
    max_seq_len = max_len;
    current_len = 0;
    
    key_caches.resize(layers);
    value_caches.resize(layers);
    size_t per_layer = (size_t)kv_heads * max_len * hdim;
    for (uint32_t i = 0; i < layers; i++) {
        key_caches[i] = new float[per_layer]();
        value_caches[i] = new float[per_layer]();
    }
}

void KVCache::reset() {
    current_len = 0;
    for (uint32_t i = 0; i < n_layers; i++) {
        size_t per_layer = (size_t)n_kv_heads * max_seq_len * head_dim;
        memset(key_caches[i], 0, per_layer * sizeof(float));
        memset(value_caches[i], 0, per_layer * sizeof(float));
    }
}

KVCache::~KVCache() {
    for (auto p : key_caches) delete[] p;
    for (auto p : value_caches) delete[] p;
}

void attention_with_cache_f32(
    const float* q,
    KVCache& cache,
    uint32_t layer_idx,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float scale,
    const float* o_weight,
    float* output)
{
    uint32_t seq_len = cache.current_len + 1;
    uint32_t n_groups = n_heads / n_kv_heads;
    uint32_t t_q = seq_len - 1;  // current token position
    
    // Store new K,V into cache
    // q is actually QKV output for the new token
    // We need K and V separately from the attention layer
    // This function assumes K and V are already computed and stored
    
    // For now: compute attention over all cached keys
    uint32_t hidden_size = n_heads * head_dim;
    memset(output, 0, hidden_size * sizeof(float));
    
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = h / n_groups;
        
        // Compute scores for all cached positions
        float* scores = new float[seq_len];
        float max_score = -1e30f;
        
        for (uint32_t t_k = 0; t_k < seq_len; t_k++) {
            float dot = 0.0f;
            const float* qh = q + h * head_dim;
            const float* kh = cache.key_caches[layer_idx] + 
                (kv_h * cache.max_seq_len + t_k) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
            scores[t_k] = dot * scale;
            if (scores[t_k] > max_score) max_score = scores[t_k];
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (uint32_t t_k = 0; t_k < seq_len; t_k++) {
            scores[t_k] = expf(scores[t_k] - max_score);
            sum_exp += scores[t_k];
        }
        for (uint32_t t_k = 0; t_k < seq_len; t_k++) scores[t_k] /= sum_exp;
        
        // Weighted sum
        float* out_h = output + h * head_dim;
        for (uint32_t t_k = 0; t_k < seq_len; t_k++) {
            const float* vh = cache.value_caches[layer_idx] + 
                (kv_h * cache.max_seq_len + t_k) * head_dim;
            float w = scores[t_k];
            for (uint32_t d = 0; d < head_dim; d++) out_h[d] += w * vh[d];
        }
        delete[] scores;
    }
    
    // Output projection
    float* merged = new float[hidden_size];
    for (uint32_t h = 0; h < n_heads; h++) {
        memcpy(merged + h * head_dim, output + h * head_dim, head_dim * sizeof(float));
    }
    for (uint32_t j = 0; j < hidden_size; j++) {
        float s = 0.0f;
        for (uint32_t k = 0; k < hidden_size; k++) {
            s += merged[k] * o_weight[j * hidden_size + k];
        }
        output[j] = s;
    }
    delete[] merged;
}
