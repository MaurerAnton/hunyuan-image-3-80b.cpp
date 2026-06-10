#include "hunyuan_image_gen.h"
#include "hunyuan_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

static std::vector<float> load_bin(const std::string& path) {
    std::vector<float> d;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); return d; }
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / 4; fseek(f, 0, SEEK_SET);
    d.resize(n); fread(d.data(), 4, n, f); fclose(f); return d;
}

static float rms(const float* x, size_t n) {
    double s = 0; for (size_t i = 0; i < n; i++) s += (double)x[i]*x[i];
    return (float)sqrt(s / n);
}

static float correlation(const float* a, const float* b, size_t n) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0;
    for (size_t i=0;i<n;i++){sa+=a[i];sb+=b[i];sab+=(double)a[i]*b[i];sa2+=(double)a[i]*a[i];sb2+=(double)b[i]*b[i];}
    double ma=sa/n,mb=sb/n,cov=sab-n*ma*mb,va=sa2-n*ma*ma,vb=sb2-n*mb*mb;
    return va<1e-30||vb<1e-30?0:(float)(cov/sqrt(va*vb));
}

void test_timestep(const char* dir) {
    printf("=== TimestepEmbedder ===\n");
    
    auto w0 = load_bin((std::string(dir)+"/timestep_timestep_emb_mlp_0_weight.bin").c_str());
    auto b0 = load_bin((std::string(dir)+"/timestep_timestep_emb_mlp_0_bias.bin").c_str());
    auto w2 = load_bin((std::string(dir)+"/timestep_timestep_emb_mlp_2_weight.bin").c_str());
    auto b2 = load_bin((std::string(dir)+"/timestep_timestep_emb_mlp_2_bias.bin").c_str());
    auto t_in = load_bin((std::string(dir)+"/timestep_input.bin").c_str());
    auto py_out = load_bin((std::string(dir)+"/timestep_py.bin").c_str());
    
    if (w0.empty()) { printf("  SKIP: no weights\n"); return; }
    
    uint32_t freq_dim = 256, hidden = w0.size() / freq_dim;
    uint32_t out_size = w2.size() / hidden, batch = t_in.size();
    
    TimestepEmbedderWeights tw;
    tw.frequency_embedding_size = freq_dim; tw.hidden_size = hidden;
    tw.out_size = out_size; tw.max_period = 10000.0f;
    tw.mlp_0_weight = std::move(w0); tw.mlp_0_bias = std::move(b0);
    tw.mlp_2_weight = std::move(w2); tw.mlp_2_bias = std::move(b2);
    
    std::vector<float> cpp_out(batch * out_size);
    timestep_embedding_f32(t_in.data(), batch, tw, cpp_out.data());
    
    float cr = rms(cpp_out.data(), cpp_out.size());
    float pr = rms(py_out.data(), py_out.size());
    float corr = correlation(cpp_out.data(), py_out.data(), cpp_out.size());
    printf("  C++ RMS=%.6f  Py RMS=%.6f  corr=%.10f  %s\n",
           cr, pr, corr, corr > 0.9999 ? "BYTE-PERFECT" : corr > 0.99 ? "CLOSE" : "MISMATCH");
}

void test_unet_down(const char* dir) {
    printf("=== UNetDown (patch_embed) ===\n");
    
    auto x = load_bin((std::string(dir)+"/unet_down_input.bin").c_str());
    auto py_out = load_bin((std::string(dir)+"/unet_down_output_py.bin").c_str());
    
    if (x.empty()) { printf("  SKIP: no input\n"); return; }
    if (py_out.empty()) { printf("  SKIP: no Python reference output (run verify_image_gen.py first)\n"); return; }
    
    // TODO: load UNetDown weights and compare
    printf("  TODO: implement C++ weight loading from .bin files\n");
}

void test_unet_up(const char* dir) {
    printf("=== UNetUp (final_layer) ===\n");
    
    auto x = load_bin((std::string(dir)+"/unet_up_input.bin").c_str());
    auto py_out = load_bin((std::string(dir)+"/unet_up_output_py.bin").c_str());
    
    if (x.empty()) { printf("  SKIP: no input\n"); return; }
    if (py_out.empty()) { printf("  SKIP: no Python reference output (run verify_image_gen.py first)\n"); return; }
    
    // TODO: load UNetUp weights and compare
    printf("  TODO: implement C++ weight loading from .bin files\n");
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "test_image_gen";
    printf("Testing image gen components with ref data from: %s\n\n", dir);
    test_timestep(dir);
    test_unet_down(dir);
    test_unet_up(dir);
    return 0;
}
