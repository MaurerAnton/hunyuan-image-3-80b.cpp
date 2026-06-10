#include "hunyuan_vae.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ============================================================
// SiLU/Swish activation
// ============================================================

static inline float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

// ============================================================
// Conv3d (simplified for T=1 image case)
// ============================================================

void conv3d_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const Conv3dWeights& w, float* out)
{
    uint32_t T_out = (T + 2*w.pad_t - w.kT) / w.stride_t + 1;
    uint32_t H_out = (H + 2*w.pad_h - w.kH) / w.stride_h + 1;
    uint32_t W_out = (W + 2*w.pad_w - w.kW) / w.stride_w + 1;
    
    memset(out, 0, B * w.out_ch * T_out * H_out * W_out * sizeof(float));
    
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t oc = 0; oc < w.out_ch; oc++) {
            for (uint32_t ot = 0; ot < T_out; ot++) {
                for (uint32_t oh = 0; oh < H_out; oh++) {
                    for (uint32_t ow = 0; ow < W_out; ow++) {
                        float sum = w.bias.empty() ? 0.0f : w.bias[oc];
                        for (uint32_t ic = 0; ic < w.in_ch; ic++) {
                            for (uint32_t kt = 0; kt < w.kT; kt++) {
                                int it = ot * w.stride_t + kt - w.pad_t;
                                if (it < 0 || it >= (int)T) continue;
                                for (uint32_t kh = 0; kh < w.kH; kh++) {
                                    int ih = oh * w.stride_h + kh - w.pad_h;
                                    if (ih < 0 || ih >= (int)H) continue;
                                    for (uint32_t kw = 0; kw < w.kW; kw++) {
                                        int iw = ow * w.stride_w + kw - w.pad_w;
                                        if (iw < 0 || iw >= (int)W) continue;
                                        float xv = x[((((b * w.in_ch + ic) * T + it) * H + ih) * W + iw)];
                                        float wv = w.weight[((((oc * w.in_ch + ic) * w.kT + kt) * w.kH + kh) * w.kW + kw)];
                                        sum += xv * wv;
                                    }
                                }
                            }
                        }
                        out[((((b * w.out_ch + oc) * T_out + ot) * H_out + oh) * W_out + ow)] = sum;
                    }
                }
            }
        }
    }
    
    // Add bias if present
    if (!w.bias.empty()) {
        // Bias already added in loop above, nothing to do
    }
}

// ============================================================
// GroupNorm for 5D tensors [B, C, T, H, W]
// ============================================================

void group_norm_5d_f32(
    const float* x, uint32_t B, uint32_t C, uint32_t T, uint32_t H, uint32_t W,
    uint32_t num_groups, const float* weight, const float* bias,
    float eps, float* out)
{
    uint32_t ch_per_group = C / num_groups;
    uint32_t spatial = T * H * W;
    
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t g = 0; g < num_groups; g++) {
            uint32_t ch_start = g * ch_per_group;
            uint32_t n = ch_per_group * spatial;
            
            // Compute mean and variance
            float mean = 0, var = 0;
            float* vals = new float[n];
            uint32_t idx = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                for (uint32_t t = 0; t < T; t++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            float v = x[((((b * C + c) * T + t) * H + h) * W + w)];
                            vals[idx++] = v;
                            mean += v;
                        }
                    }
                }
            }
            mean /= n;
            idx = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                for (uint32_t t = 0; t < T; t++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            float d = vals[idx++] - mean;
                            var += d * d;
                        }
                    }
                }
            }
            var = var / n;
            float inv_std = 1.0f / sqrtf(var + eps);
            
            // Normalize + scale + shift
            idx = 0;
            for (uint32_t c = ch_start; c < ch_start + ch_per_group; c++) {
                float wc = weight[c], bc = bias[c];
                for (uint32_t t = 0; t < T; t++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            float normed = (vals[idx++] - mean) * inv_std;
                            out[((((b * C + c) * T + t) * H + h) * W + w)] = normed * wc + bc;
                        }
                    }
                }
            }
            delete[] vals;
        }
    }
}

// ============================================================
// ResnetBlock (VAE style, 3D)
// ============================================================

void resnet_block_3d_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const ResnetBlock3DWeights& w, float* out)
{
    uint32_t in_ch = w.in_channels;
    uint32_t out_ch = w.out_channels;
    
    // norm1 → swish → conv1
    float* h = new float[B * in_ch * T * H * W];
    group_norm_5d_f32(x, B, in_ch, T, H, W, 32, w.norm1_weight.data(), w.norm1_bias.data(), 1e-6f, h);
    for (uint32_t i = 0; i < B * in_ch * T * H * W; i++) h[i] = silu(h[i]);
    
    float* conv1_out = new float[B * out_ch * T * H * W];
    conv3d_f32(h, B, T, H, W, w.conv1, conv1_out);
    delete[] h;
    
    // norm2 → swish → conv2
    h = new float[B * out_ch * T * H * W];
    group_norm_5d_f32(conv1_out, B, out_ch, T, H, W, 32, w.norm2_weight.data(), w.norm2_bias.data(), 1e-6f, h);
    delete[] conv1_out;
    for (uint32_t i = 0; i < B * out_ch * T * H * W; i++) h[i] = silu(h[i]);
    
    float* conv2_out = new float[B * out_ch * T * H * W];
    conv3d_f32(h, B, T, H, W, w.conv2, conv2_out);
    delete[] h;
    
    // Shortcut
    if (in_ch == out_ch) {
        for (uint32_t i = 0; i < B * out_ch * T * H * W; i++)
            out[i] = conv2_out[i] + x[i];
    } else {
        float* shortcut = new float[B * out_ch * T * H * W];
        conv3d_f32(x, B, T, H, W, w.shortcut_conv, shortcut);
        for (uint32_t i = 0; i < B * out_ch * T * H * W; i++)
            out[i] = conv2_out[i] + shortcut[i];
        delete[] shortcut;
    }
    delete[] conv2_out;
}

// ============================================================
// AttnBlock (self-attention over spatial dims)
// ============================================================

void attn_block_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const AttnBlockWeights& w, float* out)
{
    uint32_t C = w.in_channels;
    uint32_t N = T * H * W;  // spatial sequence length
    
    // norm
    float* normed = new float[B * C * T * H * W];
    group_norm_5d_f32(x, B, C, T, H, W, 32, w.norm_weight.data(), w.norm_bias.data(), 1e-6f, normed);
    
    // Q, K, V via 1×1×1 conv
    float *q = new float[B * C * N], *k = new float[B * C * N], *v = new float[B * C * N];
    conv3d_f32(normed, B, T, H, W, w.q_conv, q);  // q is [B, C, T, H, W]
    conv3d_f32(normed, B, T, H, W, w.k_conv, k);
    conv3d_f32(normed, B, T, H, W, w.v_conv, v);
    delete[] normed;
    
    // Rearrange: [B, C, T, H, W] → [B, 1, N, C]
    // Q: each head_dim=C for single-head attention
    float* q_r = new float[B * N * C];  // [B, N, C]
    float* k_r = new float[B * N * C];
    float* v_r = new float[B * N * C];
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t n = 0; n < N; n++) {
            uint32_t t = n / (H * W);
            uint32_t hw = n % (H * W);
            uint32_t h = hw / W, w_i = hw % W;
            for (uint32_t c = 0; c < C; c++) {
                q_r[(b * N + n) * C + c] = q[((((b * C + c) * T + t) * H + h) * W + w_i)];
                k_r[(b * N + n) * C + c] = k[((((b * C + c) * T + t) * H + h) * W + w_i)];
                v_r[(b * N + n) * C + c] = v[((((b * C + c) * T + t) * H + h) * W + w_i)];
            }
        }
    }
    delete[] q; delete[] k; delete[] v;
    
    // Scaled dot-product attention
    float scale = 1.0f / sqrtf((float)C);
    float* attn_out = new float[B * N * C]();
    
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t ni = 0; ni < N; ni++) {
            // Compute scores for all keys
            float* scores = new float[N];
            float max_s = -1e30f;
            for (uint32_t nj = 0; nj < N; nj++) {
                float dot = 0;
                for (uint32_t c = 0; c < C; c++)
                    dot += q_r[(b * N + ni) * C + c] * k_r[(b * N + nj) * C + c];
                scores[nj] = dot * scale;
                if (scores[nj] > max_s) max_s = scores[nj];
            }
            float sum_e = 0;
            for (uint32_t nj = 0; nj < N; nj++) {
                scores[nj] = expf(scores[nj] - max_s);
                sum_e += scores[nj];
            }
            for (uint32_t nj = 0; nj < N; nj++) scores[nj] /= sum_e;
            
            for (uint32_t c = 0; c < C; c++) {
                float s = 0;
                for (uint32_t nj = 0; nj < N; nj++)
                    s += scores[nj] * v_r[(b * N + nj) * C + c];
                attn_out[(b * N + ni) * C + c] = s;
            }
            delete[] scores;
        }
    }
    delete[] q_r; delete[] k_r; delete[] v_r;
    
    // Rearrange back: [B, N, C] → [B, C, T, H, W]
    float* attn_3d = new float[B * C * T * H * W];
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t n = 0; n < N; n++) {
            uint32_t t = n / (H * W);
            uint32_t hw = n % (H * W);
            uint32_t h = hw / W, w_i = hw % W;
            for (uint32_t c = 0; c < C; c++) {
                attn_3d[((((b * C + c) * T + t) * H + h) * W + w_i)] = attn_out[(b * N + n) * C + c];
            }
        }
    }
    delete[] attn_out;
    
    // proj_out (1×1×1 conv) → residual
    float* proj = new float[B * C * T * H * W];
    conv3d_f32(attn_3d, B, T, H, W, w.proj_out, proj);
    delete[] attn_3d;
    
    for (uint32_t i = 0; i < B * C * T * H * W; i++)
        out[i] = x[i] + proj[i];
    delete[] proj;
}

// ============================================================
// UpsampleDCAE
// ============================================================

void upsample_dcae_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const UpsampleDCAEWeights& w, float* out)
{
    uint32_t r1 = w.add_temporal_upsample ? 2 : 1;
    uint32_t r2 = 2, r3 = 2;
    uint32_t factor = r1 * r2 * r3;
    uint32_t T_out = T * r1;
    uint32_t H_out = H * r2;
    uint32_t W_out = W * r3;
    uint32_t out_ch = w.out_channels;
    uint32_t in_ch = w.in_channels;
    
    // conv
    uint32_t conv_out_ch = out_ch * factor;
    Conv3dWeights conv_w = w.conv;
    conv_w.out_ch = conv_out_ch;
    float* conv_out = new float[B * conv_out_ch * T * H * W];
    conv3d_f32(x, B, T, H, W, conv_w, conv_out);
    
    // Rearrange: [B, r1*r2*r3*out_ch, T, H, W] → [B, out_ch, T*r1, H*r2, W*r3]
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t oc = 0; oc < out_ch; oc++) {
            for (uint32_t t = 0; t < T; t++) {
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w_i = 0; w_i < W; w_i++) {
                        for (uint32_t rt = 0; rt < r1; rt++) {
                            for (uint32_t rh = 0; rh < r2; rh++) {
                                for (uint32_t rw = 0; rw < r3; rw++) {
                                    uint32_t src_c = (rt * r2 * r3 + rh * r3 + rw) * out_ch + oc;
                                    float v = conv_out[((((b * conv_out_ch + src_c) * T + t) * H + h) * W + w_i)];
                                    out[((((b * out_ch + oc) * T_out + t*r1+rt) * H_out + h*r2+rh) * W_out + w_i*r3+rw)] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] conv_out;
    
    // Shortcut: x.repeat_interleave(repeats, dim=1) → rearrange → mean pool
    uint32_t repeats = factor * out_ch / in_ch;
    // repeat_interleave: x becomes [B, in_ch*repeats, T, H, W]
    // rearrange: [B, r1*r2*r3*out_ch, T, H, W] → [B, out_ch, T*r1, H*r2, W*r3]
    // then mean pool over groups of size: group_size = factor * in_ch / out_ch
    uint32_t group_size = factor * in_ch / out_ch;
    
    float* sc = new float[B * out_ch * T_out * H_out * W_out]();
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t oc = 0; oc < out_ch; oc++) {
            for (uint32_t t = 0; t < T; t++) {
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w_i = 0; w_i < W; w_i++) {
                        float xv = x[((((b * in_ch + (oc % in_ch)) * T + t) * H + h) * W + w_i)];
                        for (uint32_t rt = 0; rt < r1; rt++) {
                            for (uint32_t rh = 0; rh < r2; rh++) {
                                for (uint32_t rw = 0; rw < r3; rw++) {
                                    sc[((((b * out_ch + oc) * T_out + t*r1+rt) * H_out + h*r2+rh) * W_out + w_i*r3+rw)] += xv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Mean (divide by group_size)
    for (uint32_t i = 0; i < B * out_ch * T_out * H_out * W_out; i++)
        sc[i] /= (float)group_size;
    
    // Add shortcut to output
    for (uint32_t i = 0; i < B * out_ch * T_out * H_out * W_out; i++)
        out[i] += sc[i];
    delete[] sc;
}

// ============================================================
// VAE Decoder Forward
// ============================================================

void vae_decoder_forward_f32(
    const float* z, uint32_t B, uint32_t H, uint32_t W,
    const VAEDecoderWeights& w, float* out)
{
    uint32_t T = 1;  // Images: T=1
    uint32_t z_ch = w.z_channels;   // 32
    uint32_t block_in = w.block_out_channels[0];  // 128
    
    // conv_in: z [B,32,1,H,W] → [B,128,1,H,W] + z.repeat(4)
    uint32_t repeats = block_in / z_ch;  // 4
    float* h = new float[B * block_in * T * H * W];
    conv3d_f32(z, B, T, H, W, w.conv_in, h);
    
    // Add z repeated
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < block_in; c++) {
            uint32_t zc = c % z_ch;
            for (uint32_t thw = 0; thw < T * H * W; thw++) {
                float zv = z[((b * z_ch + zc) * T * H * W) + thw];
                h[((b * block_in + c) * T * H * W) + thw] += zv;
            }
        }
    }
    
    // mid: block_1 → attn_1 → block_2
    float* h2 = new float[B * block_in * T * H * W];
    resnet_block_3d_f32(h, B, T, H, W, w.mid_block_1, h2);
    delete[] h;
    
    h = new float[B * block_in * T * H * W];
    attn_block_f32(h2, B, T, H, W, w.mid_attn_1, h);
    delete[] h2;
    
    h2 = new float[B * block_in * T * H * W];
    resnet_block_3d_f32(h, B, T, H, W, w.mid_block_2, h2);
    delete[] h;
    h = h2;
    
    uint32_t cur_ch = block_in;
    
    // up levels
    for (size_t level = 0; level < w.up_levels.size(); level++) {
        auto& ul = w.up_levels[level];
        
        for (size_t bi = 0; bi < ul.blocks.size(); bi++) {
            uint32_t out_ch_block = ul.blocks[bi].out_channels;
            float* next = new float[B * out_ch_block * T * H * W];
            resnet_block_3d_f32(h, B, T, H, W, ul.blocks[bi], next);
            delete[] h;
            h = next;
            cur_ch = out_ch_block;
        }
        
        if (ul.has_upsample) {
            uint32_t T_new = T * (ul.upsample.add_temporal_upsample ? 2 : 1);
            uint32_t H_new = H * 2;
            uint32_t W_new = W * 2;
            uint32_t out_ch_up = ul.upsample.out_channels;
            
            float* up_out = new float[B * out_ch_up * T_new * H_new * W_new];
            upsample_dcae_f32(h, B, T, H, W, ul.upsample, up_out);
            delete[] h;
            h = up_out;
            T = T_new; H = H_new; W = W_new;
            cur_ch = out_ch_up;
        }
    }
    
    // norm_out → swish → conv_out
    float* normed = new float[B * cur_ch * T * H * W];
    group_norm_5d_f32(h, B, cur_ch, T, H, W, 32, 
                      w.norm_out_weight.data(), w.norm_out_bias.data(), 1e-6f, normed);
    delete[] h;
    for (uint32_t i = 0; i < B * cur_ch * T * H * W; i++) normed[i] = silu(normed[i]);
    
    conv3d_f32(normed, B, T, H, W, w.conv_out, out);
    delete[] normed;
}

// ============================================================
// Weight loading (stub — needs safetensors integration)
// ============================================================

bool VAEDecoderWeights::load_from_shards(const std::string&, const std::string&) {
    // TODO: Implement VAE weight loading from safetensors shards
    // This requires reading shards 31 and 32
    fprintf(stderr, "VAE weight loading not yet implemented\n");
    return false;
}
