#include "hunyuan_gguf_loader.h"
#include <cstring>
#include <cstdlib>
#include <cmath>

// ============================================================
// Dequantization helpers
// ============================================================

void HunyuanGGUFLoader::dequantize_q8_0(const void* src, float* dst, size_t n) {
    // Q8_0 block: 32 floats, block size = 2 + 32 = 34 bytes
    // Layout: [d (f16, 2 bytes)] [q (int8, 32 bytes)]
    const uint8_t* data = (const uint8_t*)src;
    size_t n_blocks = n / 32;
    size_t block_size = 34;
    
    for (size_t b = 0; b < n_blocks; b++) {
        // Read scale (f16)
        uint16_t d_half = *((const uint16_t*)(data + b * block_size));
        // F16 to F32 conversion
        uint32_t sign = (d_half & 0x8000) << 16;
        uint32_t exp = (d_half >> 10) & 0x1F;
        uint32_t mant = d_half & 0x3FF;
        float d;
        if (exp == 0) {
            uint32_t bits = sign;
            memcpy(&d, &bits, sizeof(float));
        } else {
            exp = exp - 15 + 127;
            uint32_t bits = sign | (exp << 23) | (mant << 13);
            memcpy(&d, &bits, sizeof(float));
        }
        
        const int8_t* q = (const int8_t*)(data + b * block_size + 2);
        for (size_t i = 0; i < 32 && b * 32 + i < n; i++) {
            dst[b * 32 + i] = (float)q[i] * d;
        }
    }
}

void HunyuanGGUFLoader::dequantize_q4_0(const void* src, float* dst, size_t n) {
    // Q4_0 block: 32 floats, block size = 2 + 16 = 18 bytes
    // Layout: [d (f16, 2 bytes)] [q (uint4 pairs, 16 bytes)]
    const uint8_t* data = (const uint8_t*)src;
    size_t n_blocks = n / 32;
    size_t block_size = 18;
    
    for (size_t b = 0; b < n_blocks; b++) {
        uint16_t d_half = *((const uint16_t*)(data + b * block_size));
        uint32_t sign = (d_half & 0x8000) << 16;
        uint32_t exp = (d_half >> 10) & 0x1F;
        uint32_t mant = d_half & 0x3FF;
        float d;
        if (exp == 0) {
            uint32_t bits = sign;
            memcpy(&d, &bits, sizeof(float));
        } else {
            exp = exp - 15 + 127;
            uint32_t bits = sign | (exp << 23) | (mant << 13);
            memcpy(&d, &bits, sizeof(float));
        }
        
        const uint8_t* q = data + b * block_size + 2;
        for (size_t i = 0; i < 32 && b * 32 + i < n; i++) {
            uint8_t val = q[i / 2];
            int8_t qi = (i & 1) ? (val >> 4) : (val & 0xF);
            // Q4_0: values are stored as 0..15, representing -8..7
            qi = (qi & 0x8) ? qi - 16 : qi;
            dst[b * 32 + i] = (float)qi * d;
        }
    }
}

// ============================================================
// File operations
// ============================================================

HunyuanGGUFLoader::~HunyuanGGUFLoader() {
    close();
}

void HunyuanGGUFLoader::close() {
    if (current_file_.ctx) {
        gguf_free(current_file_.ctx);
        current_file_.ctx = nullptr;
    }
    if (current_file_.fp) {
        fclose(current_file_.fp);
        current_file_.fp = nullptr;
    }
}

bool HunyuanGGUFLoader::open_global(const std::string& path) {
    close();
    
    gguf_init_params params = {};
    params.no_alloc = false;
    
    current_file_.ctx = gguf_init_from_file(path.c_str(), params);
    if (!current_file_.ctx) {
        fprintf(stderr, "Failed to open GGUF: %s\n", path.c_str());
        return false;
    }
    
    current_file_.path = path;
    current_file_.fp = fopen(path.c_str(), "rb");
    current_file_.data_offset = gguf_get_data_offset(current_file_.ctx);
    
    // Load all tensors from global file into memory (they're relatively small)
    int64_t n_tensors = gguf_get_n_tensors(current_file_.ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(current_file_.ctx, i);
        std::string sname(name);
        
        std::vector<float> data;
        if (load_tensor(sname, data)) {
            global_tensors_[sname] = std::move(data);
        }
    }
    
    printf("GGUF global: loaded %zu tensors\n", global_tensors_.size());
    return true;
}

bool HunyuanGGUFLoader::open_layer(const std::string& path) {
    close();
    
    gguf_init_params params = {};
    params.no_alloc = false;
    
    current_file_.ctx = gguf_init_from_file(path.c_str(), params);
    if (!current_file_.ctx) {
        fprintf(stderr, "Failed to open GGUF: %s\n", path.c_str());
        return false;
    }
    
    current_file_.path = path;
    current_file_.fp = fopen(path.c_str(), "rb");
    current_file_.data_offset = gguf_get_data_offset(current_file_.ctx);
    
    printf("GGUF layer: opened %s\n", path.c_str());
    return true;
}

bool HunyuanGGUFLoader::has_tensor(const std::string& name) const {
    if (!current_file_.ctx) return false;
    return gguf_find_tensor(current_file_.ctx, name.c_str()) >= 0;
}

bool HunyuanGGUFLoader::load_tensor(const std::string& name, std::vector<float>& dst) {
    if (!current_file_.ctx || !current_file_.fp) return false;
    
    int64_t idx = gguf_find_tensor(current_file_.ctx, name.c_str());
    if (idx < 0) return false;
    
    size_t offset = gguf_get_tensor_offset(current_file_.ctx, idx);
    size_t size = gguf_get_tensor_size(current_file_.ctx, idx);
    ggml_type type = gguf_get_tensor_type(current_file_.ctx, idx);
    
    return read_tensor_data(name, type, offset, size, dst);
}

bool HunyuanGGUFLoader::read_tensor_data(
    const std::string& name, ggml_type type,
    size_t offset, size_t nbytes,
    std::vector<float>& dst)
{
    // Calculate element count
    // F32: 4 bytes/elem, F16: 2 bytes/elem
    // Q8_0: 34 bytes per 32 elems, Q4_0: 18 bytes per 32 elems
    size_t nelem = 0;
    switch (type) {
        case GGML_TYPE_F32:  nelem = nbytes / 4; break;
        case GGML_TYPE_F16:  nelem = nbytes / 2; break;
        case GGML_TYPE_Q8_0: nelem = nbytes / 34 * 32; break;
        case GGML_TYPE_Q4_0: nelem = nbytes / 18 * 32; break;
        default:
            fprintf(stderr, "Unsupported GGML type %d for tensor %s\n", type, name.c_str());
            return false;
    }
    
    dst.resize(nelem);
    
    // Read raw data
    std::vector<uint8_t> raw(nbytes);
    fseek(current_file_.fp, current_file_.data_offset + offset, SEEK_SET);
    if (fread(raw.data(), 1, nbytes, current_file_.fp) != nbytes) {
        fprintf(stderr, "Failed to read tensor %s\n", name.c_str());
        return false;
    }
    
    // Convert to F32
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst.data(), raw.data(), nbytes);
            break;
        case GGML_TYPE_F16: {
            uint16_t* src16 = (uint16_t*)raw.data();
            for (size_t i = 0; i < nelem; i++) {
                uint16_t h = src16[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h & 0x7C00) >> 10;
                uint32_t mant = h & 0x03FF;
                uint32_t bits;
                if (exp == 0) bits = sign;
                else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
                else { exp = exp - 15 + 127; bits = sign | (exp << 23) | (mant << 13); }
                memcpy(&dst[i], &bits, sizeof(float));
            }
            break;
        }
        case GGML_TYPE_Q8_0:
            dequantize_q8_0(raw.data(), dst.data(), nelem);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_q4_0(raw.data(), dst.data(), nelem);
            break;
        default:
            return false;
    }
    
    return true;
}

// ============================================================
// Layer weights loading from GGUF
// ============================================================

bool HunyuanGGUFLoader::load_layer_weights(int layer_idx, DecoderLayerWeights& w) {
    if (!current_file_.ctx) return false;
    
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "blk.%d.", layer_idx);
    std::string pfx(prefix);
    
    auto load_one = [this](const std::string& name, std::vector<float>& dst) -> bool {
        if (!has_tensor(name)) {
            // Try alternative name (from safetensors convention)
            return false;
        }
        return load_tensor(name, dst);
    };
    
    // Note: GGUF stores weights transposed [in, out] for ggml convention
    // But our C++ manual loops expect [out, in].
    // We need to transpose back when loading from GGUF.
    // For now, skip this — just load raw data.
    
    // Norm weights (1D, no transpose needed)
    load_one(pfx + "attn_norm.weight", w.input_ln_weight);
    load_one(pfx + "ffn_norm.weight", w.post_attn_ln_weight);
    
    // Attention
    load_one(pfx + "attn_qkv.weight", w.attn.qkv_weight);
    load_one(pfx + "attn_output.weight", w.attn.o_weight);
    load_one(pfx + "attn_q_norm.weight", w.attn.q_norm_weight);
    load_one(pfx + "attn_k_norm.weight", w.attn.k_norm_weight);
    
    // MoE shared
    load_one(pfx + "ffn_shared_gate_up.weight", w.moe.shared_gate_up);
    load_one(pfx + "ffn_shared_down.weight", w.moe.shared_down);
    
    // Router
    load_one(pfx + "ffn_gate.weight", w.moe.router_weight);
    
    // Experts
    uint32_t num_experts = get_metadata_u32("hunyuan.num_experts", 64);
    w.moe.num_experts = num_experts;
    w.moe.experts.resize(num_experts);
    for (uint32_t e = 0; e < num_experts; e++) {
        char ebuf[128];
        snprintf(ebuf, sizeof(ebuf), "%sffn_expert_%d_gate_up.weight", prefix, e);
        load_one(std::string(ebuf), w.moe.experts[e].gate_up);
        snprintf(ebuf, sizeof(ebuf), "%sffn_expert_%d_down.weight", prefix, e);
        load_one(std::string(ebuf), w.moe.experts[e].down);
    }
    
    printf("GGUF: loaded layer %d (%u experts)\n", layer_idx, num_experts);
    return true;
}

// ============================================================
// Metadata access
// ============================================================

uint32_t HunyuanGGUFLoader::get_metadata_u32(const std::string& key, uint32_t default_val) const {
    if (!current_file_.ctx) return default_val;
    int64_t idx = gguf_find_key(current_file_.ctx, key.c_str());
    if (idx < 0) return default_val;
    return gguf_get_val_u32(current_file_.ctx, idx);
}

float HunyuanGGUFLoader::get_metadata_f32(const std::string& key, float default_val) const {
    if (!current_file_.ctx) return default_val;
    int64_t idx = gguf_find_key(current_file_.ctx, key.c_str());
    if (idx < 0) return default_val;
    return gguf_get_val_f32(current_file_.ctx, idx);
}
