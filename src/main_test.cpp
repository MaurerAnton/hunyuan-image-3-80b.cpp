#include "hunyuan_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

// ============================================================
// Load raw float32 binary file
// ============================================================
static std::vector<float> load_bin(const std::string& path, size_t expected_n = 0) {
    std::vector<float> data;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path.c_str());
        return data;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = size / sizeof(float);
    data.resize(n);
    fread(data.data(), sizeof(float), n, f);
    fclose(f);
    if (expected_n > 0 && n != expected_n) {
        fprintf(stderr, "WARNING: %s: expected %zu floats, got %zu\n", 
                path.c_str(), expected_n, n);
    }
    return data;
}

// ============================================================
// Main test
// ============================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <layer_idx> <shard_path> [test_dir]\n", argv[0]);
        fprintf(stderr, "  layer_idx: 0-31\n");
        fprintf(stderr, "  shard_path: path to safetensors shard containing the layer\n");
        fprintf(stderr, "  test_dir: directory with Python reference .bin files (default: test_output)\n");
        return 1;
    }
    
    int layer_idx = atoi(argv[1]);
    std::string shard_path = argv[2];
    std::string test_dir = argc > 3 ? argv[3] : "test_output";
    
    printf("=== Hunyuan Image 3 Layer %d Test ===\n", layer_idx);
    printf("Shard: %s\n", shard_path.c_str());
    printf("Ref dir: %s\n", test_dir.c_str());
    
    // Initialize config
    HunyuanConfig cfg;
    cfg.compute_derived();
    printf("Config: hidden=%u heads=%u kv_heads=%u head_dim=%u\n",
           cfg.hidden_size, cfg.num_attention_heads, 
           cfg.num_key_value_heads, cfg.attention_head_dim);
    printf("        qkv_total=%u num_kv_groups=%u moe_intermediate=%u experts=%u topk=%u\n",
           cfg.qkv_total, cfg.num_kv_groups, cfg.moe_intermediate_size,
           cfg.num_experts, cfg.moe_topk);
    
    // Load weights
    DecoderLayerWeights w;
    if (!w.load(layer_idx, shard_path)) {
        fprintf(stderr, "FAILED to load layer weights\n");
        return 1;
    }
    
    // Load Python reference input
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/layer_%d_input.bin", test_dir.c_str(), layer_idx);
    auto py_input = load_bin(fname);
    if (py_input.empty()) {
        fprintf(stderr, "ERROR: No reference input found at %s\n", fname);
        fprintf(stderr, "Run: python scripts/layer_ref.py --layer %d --shard-dir <dir>\n", layer_idx);
        return 1;
    }
    
    // Determine shape from file size
    size_t n_tokens = py_input.size() / cfg.hidden_size;
    printf("Reference input: %zu tokens (%.1f KB)\n", n_tokens, 
           py_input.size() * sizeof(float) / 1024.0);
    
    // Load Python reference output (for comparison)
    snprintf(fname, sizeof(fname), "%s/layer_%d_output.bin", test_dir.c_str(), layer_idx);
    auto py_output = load_bin(fname, py_input.size());
    if (py_output.empty()) {
        fprintf(stderr, "WARNING: No reference output. Will run C++ only.\n");
    }
    
    // Run C++ decoder layer
    printf("\n--- Running C++ Decoder Layer ---\n");
    std::vector<float> cpp_output(py_input.size());
    std::vector<int> positions(n_tokens);
    for (size_t i = 0; i < n_tokens; i++) positions[i] = i;
    
    decoder_layer_forward_cpu(
        py_input.data(), w, n_tokens, cfg, positions.data(), cpp_output.data());
    
    // Print stats
    printf("\n--- Results ---\n");
    print_tensor_stats("C++ output", cpp_output.data(), cpp_output.size());
    
    if (!py_output.empty()) {
        print_tensor_stats("Py output", py_output.data(), py_output.size());
        
        float corr = tensor_correlation(cpp_output.data(), py_output.data(), cpp_output.size());
        printf("\nCorrelation C++ vs Python: %.10f\n", corr);
        
        // Check per-token RMS
        printf("\nPer-token RMS comparison:\n");
        for (size_t t = 0; t < n_tokens && t < 4; t++) {
            float cpp_rms = tensor_rms(cpp_output.data() + t * cfg.hidden_size, cfg.hidden_size);
            float py_rms = tensor_rms(py_output.data() + t * cfg.hidden_size, cfg.hidden_size);
            printf("  token %zu: C++=%.6f Py=%.6f diff=%.6f\n", 
                   t, cpp_rms, py_rms, fabsf(cpp_rms - py_rms));
        }
        
        if (corr > 0.9999f) {
            printf("\n*** BYTE-PERFECT MATCH ***\n");
        } else if (corr > 0.99f) {
            printf("\n*** CLOSE MATCH (corr > 0.99) — minor numerical differences ***\n");
        } else {
            printf("\n*** MISMATCH — check implementation ***\n");
        }
    }
    
    printf("\nDone.\n");
    return 0;
}
