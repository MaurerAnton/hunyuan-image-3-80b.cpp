#include "hunyuan_model.h"

// Config from JSON (stub for now)
HunyuanConfig HunyuanConfig::from_json(const std::string& path) {
    HunyuanConfig cfg;
    // TODO: parse JSON config
    cfg.compute_derived();
    return cfg;
}

// ggml graph RMSNorm (stub for now — we use pure C++ implementation)
ggml_tensor* rms_norm_ggml(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    // TODO: ggml graph implementation
    return nullptr;
}
