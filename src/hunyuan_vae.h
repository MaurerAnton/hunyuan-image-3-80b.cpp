#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// ============================================================
// VAE Decoder (AutoencoderKLConv3D) for Hunyuan Image 3
//
// Architecture:
//   z [B,32,H,W] → conv_in [128] + z.repeat(4) 
//   → mid: 2×ResnetBlock + AttnBlock
//   → up[0..4]: 3×ResnetBlock + UpsampleDCAE (×4 levels)
//   → norm_out + swish + conv_out → RGB [B,3,16H,16W]
//
// For images, temporal dimension T=1 throughout.
// Conv3d with T=1 behaves as Conv2d.
// ============================================================

// ============================================================
// Conv3d (simplified: T=1, behaves as Conv2d)
// Weight layout: [out_ch, in_ch, kT, kH, kW]
// For T=1, kT=1 or 3 with temporal padding
// ============================================================

struct Conv3dWeights {
    std::vector<float> weight;  // [out_ch, in_ch, kT, kH, kW]
    std::vector<float> bias;    // [out_ch]
    uint32_t in_ch, out_ch;
    uint32_t kT, kH, kW;
    uint32_t stride_t, stride_h, stride_w;
    uint32_t pad_t, pad_h, pad_w;
};

void conv3d_f32(
    const float* x,          // [B, in_ch, T, H, W]
    uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const Conv3dWeights& w,
    float* out);             // [B, out_ch, T_out, H_out, W_out]

// ============================================================
// GroupNorm (for VAE: num_groups=32, eps=1e-6)
// ============================================================

void group_norm_5d_f32(
    const float* x,          // [B, C, T, H, W]
    uint32_t B, uint32_t C, uint32_t T, uint32_t H, uint32_t W,
    uint32_t num_groups,
    const float* weight,     // [C]
    const float* bias,       // [C]
    float eps, float* out);

// ============================================================
// ResnetBlock (VAE style: 3D convolutions)
// norm1 → swish → conv1(3×3×3) → norm2 → swish → conv2(3×3×3) → +shortcut
// ============================================================

struct ResnetBlock3DWeights {
    uint32_t in_channels, out_channels;
    
    // norm1: GroupNorm
    std::vector<float> norm1_weight, norm1_bias;
    // conv1: Conv3d(3×3×3, pad=1)
    Conv3dWeights conv1;
    // norm2: GroupNorm
    std::vector<float> norm2_weight, norm2_bias;
    // conv2: Conv3d(3×3×3, pad=1)
    Conv3dWeights conv2;
    // shortcut (if in != out): Conv3d(1×1×1)
    bool has_shortcut = false;
    Conv3dWeights shortcut_conv;
};

void resnet_block_3d_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const ResnetBlock3DWeights& w, float* out);

// ============================================================
// AttnBlock (self-attention on spatial dimensions)
// norm → Q,K,V (1×1×1 conv) → rearrange → SDPA → rearrange → proj_out → +x
// ============================================================

struct AttnBlockWeights {
    uint32_t in_channels;
    std::vector<float> norm_weight, norm_bias;
    Conv3dWeights q_conv, k_conv, v_conv;  // 1×1×1 convs
    Conv3dWeights proj_out;                // 1×1×1 conv
};

void attn_block_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const AttnBlockWeights& w, float* out);

// ============================================================
// UpsampleDCAE
// conv3d → rearrange (pixel shuffle inverse) + shortcut (mean pool)
// ============================================================

struct UpsampleDCAEWeights {
    uint32_t in_channels, out_channels;
    bool add_temporal_upsample;
    Conv3dWeights conv;  // 3×3×3 conv
};

void upsample_dcae_f32(
    const float* x, uint32_t B, uint32_t T, uint32_t H, uint32_t W,
    const UpsampleDCAEWeights& w,
    float* out);  // [B, out_ch, T*r1, H*2, W*2]

// ============================================================
// VAE Decoder
// ============================================================

struct VAEDecoderWeights {
    uint32_t z_channels = 32;          // latent channels
    uint32_t out_channels = 3;          // RGB
    std::vector<uint32_t> block_out_channels;  // [128, 256, 512, 1024, 1024]
    uint32_t num_res_blocks = 2;        // layers_per_block
    uint32_t ffactor_spatial = 16;      // total spatial upsampling
    uint32_t ffactor_temporal = 4;      // total temporal upsampling
    
    // conv_in: 32 → 128
    Conv3dWeights conv_in;
    
    // mid: 2 ResnetBlocks + AttnBlock
    ResnetBlock3DWeights mid_block_1;
    AttnBlockWeights mid_attn_1;
    ResnetBlock3DWeights mid_block_2;
    
    // up levels: each has (num_res_blocks+1) ResnetBlocks + optional UpsampleDCAE
    struct UpLevel {
        std::vector<ResnetBlock3DWeights> blocks;
        bool has_upsample = false;
        UpsampleDCAEWeights upsample;
    };
    std::vector<UpLevel> up_levels;
    
    // output: norm + conv
    std::vector<float> norm_out_weight, norm_out_bias;
    Conv3dWeights conv_out;
    
    // Load from safetensors
    bool load_from_shards(const std::string& shard_dir, const std::string& index_path);
};

void vae_decoder_forward_f32(
    const float* z,           // [B, z_channels, H, W] — latent
    uint32_t B, uint32_t H, uint32_t W,
    const VAEDecoderWeights& w,
    float* out);              // [B, 3, 16*H, 16*W] — RGB image
