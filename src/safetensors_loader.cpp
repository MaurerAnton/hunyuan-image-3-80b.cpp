#include "hunyuan_model.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>

// Minimal JSON parser for safetensors headers
// Only handles what we need: string keys, number values, arrays

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path.c_str());
        return "";
    }
    size_t size = f.tellg();
    std::string data(size, '\0');
    f.seekg(0);
    f.read(&data[0], size);
    return data;
}

// Extremely minimal JSON value extraction
static int64_t json_get_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return -1;
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;
    return strtoll(json.c_str() + pos, nullptr, 10);
}

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    size_t end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static std::vector<int64_t> json_get_int_array(const std::string& json, const std::string& key) {
    std::vector<int64_t> result;
    std::string search = "\"" + key + "\":[";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos += search.size();
    size_t end = json.find(']', pos);
    if (end == std::string::npos) return result;
    std::string arr = json.substr(pos, end - pos);
    // Parse comma-separated ints
    size_t i = 0;
    while (i < arr.size()) {
        while (i < arr.size() && (arr[i] == ' ' || arr[i] == ',')) i++;
        if (i >= arr.size()) break;
        size_t j = i;
        while (j < arr.size() && arr[j] != ',' && arr[j] != ' ') j++;
        result.push_back(strtoll(arr.c_str() + i, nullptr, 10));
        i = j;
    }
    return result;
}

// ============================================================
// Safetensors loading
// ============================================================

struct SafetensorsFile {
    std::string path;
    FILE* fp;
    size_t header_size;
    std::string header_json;
    
    bool open(const std::string& p) {
        path = p;
        fp = fopen(p.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "Cannot open safetensors: %s\n", p.c_str());
            return false;
        }
        // Read 8-byte header size
        uint64_t hs;
        if (fread(&hs, 8, 1, fp) != 1) {
            fprintf(stderr, "Failed to read header size\n");
            return false;
        }
        header_size = hs;
        // Read header JSON
        header_json.resize(header_size);
        if (fread(&header_json[0], 1, header_size, fp) != header_size) {
            fprintf(stderr, "Failed to read header\n");
            return false;
        }
        printf("Opened %s: header=%zu bytes\n", p.c_str(), header_size);
        return true;
    }
    
    bool has_tensor(const std::string& name) {
        return header_json.find("\"" + name + "\"") != std::string::npos;
    }
    
    // Get tensor shape and data offset
    bool get_tensor_info(const std::string& name, 
                         std::vector<uint32_t>& shape, 
                         size_t& data_offset,
                         size_t& data_size) {
        // Find the tensor entry: "name":{"dtype":"...","shape":[...],"data_offsets":[start,end]}
        std::string search = "\"" + name + "\":{";
        size_t pos = header_json.find(search);
        if (pos == std::string::npos) {
            fprintf(stderr, "Tensor '%s' not found in %s\n", name.c_str(), path.c_str());
            return false;
        }
        
        // Find shape array
        shape = {};
        std::string shape_search = "\"shape\":[";
        size_t sp = header_json.find(shape_search, pos);
        if (sp != std::string::npos) {
            sp += shape_search.size();
            size_t ep = header_json.find(']', sp);
            std::string arr = header_json.substr(sp, ep - sp);
            size_t i = 0;
            while (i < arr.size()) {
                while (i < arr.size() && (arr[i] == ' ' || arr[i] == ',')) i++;
                if (i >= arr.size()) break;
                size_t j = i;
                while (j < arr.size() && arr[j] != ',' && arr[j] != ' ') j++;
                shape.push_back((uint32_t)strtoul(arr.c_str() + i, nullptr, 10));
                i = j;
            }
        }
        
        // Find data_offsets
        std::string off_search = "\"data_offsets\":[";
        size_t op = header_json.find(off_search, pos);
        if (op == std::string::npos) return false;
        op += off_search.size();
        size_t comma = header_json.find(',', op);
        size_t ep = header_json.find(']', op);
        data_offset = strtoull(header_json.c_str() + op, nullptr, 10);
        data_size = strtoull(header_json.c_str() + comma + 1, nullptr, 10) - data_offset;
        
        return true;
    }
    
    // Read tensor data into float array
    bool read_tensor_f32(const std::string& name, std::vector<float>& data) {
        std::vector<uint32_t> shape;
        size_t offset, size;
        if (!get_tensor_info(name, shape, offset, size)) return false;
        
        // Check dtype
        std::string dtype_search = "\"dtype\":\"";
        size_t dp = header_json.find(dtype_search, header_json.find("\"" + name + "\""));
        std::string dtype = "F32";
        if (dp != std::string::npos) {
            dp += dtype_search.size();
            size_t de = header_json.find('"', dp);
            dtype = header_json.substr(dp, de - dp);
        }
        
        // Calculate element count
        size_t nelem = 1;
        for (auto s : shape) nelem *= s;
        
        data.resize(nelem);
        
        // Read raw data
        fseek(fp, 8 + header_size + offset, SEEK_SET);
        
        if (dtype == "F32") {
            if (fread(data.data(), sizeof(float), nelem, fp) != nelem) {
                fprintf(stderr, "Failed to read F32 tensor %s\n", name.c_str());
                return false;
            }
        } else if (dtype == "BF16") {
            // Read as uint16, convert to float32
            std::vector<uint16_t> raw(nelem);
            if (fread(raw.data(), sizeof(uint16_t), nelem, fp) != nelem) {
                fprintf(stderr, "Failed to read BF16 tensor %s\n", name.c_str());
                return false;
            }
            for (size_t i = 0; i < nelem; i++) {
                // BF16 to F32: shift left 16 bits
                uint32_t bits = ((uint32_t)raw[i]) << 16;
                memcpy(&data[i], &bits, sizeof(float));
            }
        } else if (dtype == "F16") {
            std::vector<uint16_t> raw(nelem);
            if (fread(raw.data(), sizeof(uint16_t), nelem, fp) != nelem) {
                return false;
            }
            // F16 to F32 conversion
            for (size_t i = 0; i < nelem; i++) {
                uint16_t h = raw[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h & 0x7C00) >> 10;
                uint32_t mant = (h & 0x03FF);
                uint32_t fbits;
                if (exp == 0) {
                    // Subnormal or zero
                    fbits = sign;
                } else if (exp == 31) {
                    // Infinity or NaN
                    fbits = sign | 0x7F800000 | (mant << 13);
                } else {
                    exp = exp - 15 + 127;
                    fbits = sign | (exp << 23) | (mant << 13);
                }
                memcpy(&data[i], &fbits, sizeof(float));
            }
        } else {
            fprintf(stderr, "Unsupported dtype: %s\n", dtype.c_str());
            return false;
        }
        
        // Keep PyTorch native layout [out_features, in_features]
        // Our manual matmul loops use: weight[out_idx * in_features + in_idx]
        // This matches PyTorch's row-major storage exactly — NO transpose needed.
        
        printf("  Loaded %s: dtype=%s shape=[", name.c_str(), dtype.c_str());
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) printf(", ");
            printf("%u", shape[i]);
        }
        printf("] nelem=%zu\n", nelem);
        
        return true;
    }
    
    void close() {
        if (fp) { fclose(fp); fp = nullptr; }
    };
};

// Public API: load a single tensor from a safetensors file
bool safetensors_load_tensor(const std::string& path, const std::string& name, std::vector<float>& dst) {
    SafetensorsFile sf;
    if (!sf.open(path)) return false;
    if (!sf.has_tensor(name)) { sf.close(); return false; }
    bool ok = sf.read_tensor_f32(name, dst);
    sf.close();
    return ok;
}

// ============================================================
// DecoderLayerWeights::load
// ============================================================

bool DecoderLayerWeights::load(int layer_idx, const std::string& shard_path) {
    // Support multiple shards: comma-separated or space-separated paths
    std::vector<std::string> shard_paths;
    std::string remaining = shard_path;
    while (!remaining.empty()) {
        // Skip whitespace/commas
        size_t start = remaining.find_first_not_of(" ,\t");
        if (start == std::string::npos) break;
        remaining = remaining.substr(start);
        size_t end = remaining.find_first_of(" ,\t");
        if (end == std::string::npos) {
            shard_paths.push_back(remaining);
            break;
        }
        shard_paths.push_back(remaining.substr(0, end));
        remaining = remaining.substr(end + 1);
    }
    
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", layer_idx);
    std::string pfx(prefix);
    
    // Helper: try to load tensor from any shard
    auto load_any = [&](const std::string& name, std::vector<float>& dst) -> bool {
        for (auto& sp : shard_paths) {
            SafetensorsFile sf;
            if (!sf.open(sp)) continue;
            if (sf.has_tensor(name)) {
                bool ok = sf.read_tensor_f32(name, dst);
                sf.close();
                return ok;
            }
            sf.close();
        }
        fprintf(stderr, "Tensor '%s' not found in any shard\n", name.c_str());
        return false;
    };
    
    // Norm weights
    load_any(pfx + "input_layernorm.weight", input_ln_weight);
    load_any(pfx + "post_attention_layernorm.weight", post_attn_ln_weight);
    
    // Attention
    printf("Loading attention weights...\n");
    load_any(pfx + "self_attn.qkv_proj.weight", attn.qkv_weight);
    load_any(pfx + "self_attn.o_proj.weight", attn.o_weight);
    load_any(pfx + "self_attn.query_layernorm.weight", attn.q_norm_weight);
    load_any(pfx + "self_attn.key_layernorm.weight", attn.k_norm_weight);
    
    // MoE: shared expert
    printf("Loading MoE weights...\n");
    load_any(pfx + "mlp.shared_mlp.gate_and_up_proj.weight", moe.shared_gate_up);
    load_any(pfx + "mlp.shared_mlp.down_proj.weight", moe.shared_down);
    
    // Router
    load_any(pfx + "mlp.gate.wg.weight", moe.router_weight);
    
    // Routed experts
    moe.num_experts = 64;
    moe.experts.resize(moe.num_experts);
    for (uint32_t e = 0; e < moe.num_experts; e++) {
        char ebuf[256];
        snprintf(ebuf, sizeof(ebuf), "%smlp.experts.%d.gate_and_up_proj.weight", prefix, e);
        load_any(std::string(ebuf), moe.experts[e].gate_up);
        snprintf(ebuf, sizeof(ebuf), "%smlp.experts.%d.down_proj.weight", prefix, e);
        load_any(std::string(ebuf), moe.experts[e].down);
    }
    
    printf("DecoderLayer %d loaded successfully.\n", layer_idx);
    return true;
}
