#include "hunyuan_model.h"
#include "hunyuan_image_gen.h"
#include "hunyuan_vae.h"
#include "hunyuan_moe_full.h"
#include "hunyuan_vit.h"
#include "hunyuan_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static float rand_float() { return (float)rand() / RAND_MAX; }

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: test_comp <name>\n"); return 1; }
    std::string test = argv[1];
    srand(42);
    
    if (test == "groupnorm2d") {
        uint32_t B=1, C=128, H=8, W=8;
        size_t n = B*C*H*W;
        float* x = new float[n];
        float* w = new float[C];
        float* b = new float[C];
        float* out = new float[n];
        for (size_t i = 0; i < n; i++) x[i] = rand_float()*2-1;
        for (uint32_t i = 0; i < C; i++) { w[i] = 1.0f; b[i] = 0.0f; }
        group_norm_f32(x, B, C, H, W, 32, w, b, 1e-5f, out);
        printf("GroupNorm2D [1,128,8,8]: RMS_in=%.6f RMS_out=%.6f\n", tensor_rms(x,n), tensor_rms(out,n));
        delete[] x; delete[] w; delete[] b; delete[] out;
    }
    else if (test == "groupnorm3d") {
        uint32_t B=1, C=64, T=1, H=8, W=8;
        size_t n = B*C*T*H*W;
        float* x = new float[n];
        float* w = new float[C];
        float* b = new float[C];
        float* out = new float[n];
        for (size_t i = 0; i < n; i++) x[i] = rand_float()*2-1;
        for (uint32_t i = 0; i < C; i++) { w[i] = 1.0f; b[i] = 0.0f; }
        group_norm_5d_f32(x, B, C, T, H, W, 32, w, b, 1e-6f, out);
        printf("GroupNorm3D [1,64,1,8,8]: RMS_in=%.6f RMS_out=%.6f\n", tensor_rms(x,n), tensor_rms(out,n));
        delete[] x; delete[] w; delete[] b; delete[] out;
    }
    else if (test == "conv2d") {
        uint32_t B=1, in_ch=3, H=8, W=8, out_ch=16, kH=3, kW=3;
        size_t n_in = B*in_ch*H*W, n_w = out_ch*in_ch*kH*kW, n_out = B*out_ch*H*W;
        float* x = new float[n_in];
        float* weight = new float[n_w];
        float* out = new float[n_out];
        float* bias = new float[out_ch]();
        for (size_t i = 0; i < n_in; i++) x[i] = rand_float();
        for (size_t i = 0; i < n_w; i++) weight[i] = rand_float()*0.1f;
        conv2d_f32(x, B, in_ch, H, W, weight, bias, out_ch, kH, kW, 1, 1, out);
        printf("Conv2D [%dx%dx%d -> %d]: RMS=%.6f\n", in_ch, H, W, out_ch, tensor_rms(out,n_out));
        delete[] x; delete[] weight; delete[] bias; delete[] out;
    }
    else if (test == "conv3d") {
        uint32_t B=1, in_ch=3, T=1, H=8, W=8, out_ch=16, kT=3, kH=3, kW=3;
        size_t n_in = B*in_ch*T*H*W, n_w = out_ch*in_ch*kT*kH*kW, n_out = B*out_ch*T*H*W;
        float* x = new float[n_in];
        float* weight = new float[n_w];
        float* out = new float[n_out];
        float* bias = new float[out_ch]();
        for (size_t i = 0; i < n_in; i++) x[i] = rand_float();
        for (size_t i = 0; i < n_w; i++) weight[i] = rand_float()*0.1f;
        Conv3dWeights w; w.in_ch=in_ch; w.out_ch=out_ch; w.kT=kT; w.kH=kH; w.kW=kW;
        w.stride_t=1; w.stride_h=1; w.stride_w=1; w.pad_t=1; w.pad_h=1; w.pad_w=1;
        w.weight.assign(weight, weight+n_w); w.bias.assign(bias, bias+out_ch);
        conv3d_f32(x, B, T, H, W, w, out);
        printf("Conv3D [%dx%dx%dx%d -> %d]: RMS=%.6f\n", in_ch, T, H, W, out_ch, tensor_rms(out,n_out));
        delete[] x; delete[] weight; delete[] bias; delete[] out;
    }
    else if (test == "layernorm") {
        uint32_t N=4, D=256;
        size_t n = N*D;
        float* x = new float[n];
        float* w = new float[D];
        float* b = new float[D];
        float* out = new float[n];
        for (size_t i = 0; i < n; i++) x[i] = rand_float()*2-1;
        for (uint32_t i = 0; i < D; i++) { w[i] = 1.0f; b[i] = 0.0f; }
        layer_norm_f32(x, N, D, w, b, 1e-5f, out);
        printf("LayerNorm [%dx%d]: RMS_in=%.6f RMS_out=%.6f\n", N, D, tensor_rms(x,n), tensor_rms(out,n));
        delete[] x; delete[] w; delete[] b; delete[] out;
    }
    else if (test == "softmax_topk") {
        uint32_t N=4, E=64, K=8;
        float* logits = new float[N*E];
        float* w = new float[N*K];
        int32_t* idx = new int32_t[N*K];
        for (size_t i = 0; i < N*E; i++) logits[i] = rand_float()*10-5;
        topkgating_easy(logits, N, E, K, w, idx);
        printf("Easy TopK [%dx%d k=%d]:\n", N, E, K);
        for (uint32_t i = 0; i < std::min(N,2u); i++) {
            printf("  tok %d: ", i);
            for (uint32_t k = 0; k < std::min(K,4u); k++) printf("e%d(%.3f) ", idx[i*K+k], w[i*K+k]);
            printf("\n");
        }
        delete[] logits; delete[] w; delete[] idx;
    }
    else if (test == "moe_full") {
        uint32_t N=4, E=64, K=8;
        MoERoutingConfig cfg; cfg.num_experts=E; cfg.moe_topk=K; cfg.norm_topk_prob=true;
        float* logits = new float[N*E];
        for (size_t i = 0; i < N*E; i++) logits[i] = rand_float()*10-5;
        MoEGateOutput out;
        uint32_t est_capacity = std::max(K, K * N / E) + 1;
        out.alloc(N, E, K, est_capacity, true);
        topkgating_full(logits, N, cfg, out);
        printf("Full MoE routing [%dx%d k=%d]: capacity=%d aux_loss=%.4f rate=%.3f\n",
               N, E, K, out.expert_capacity, out.aux_loss, out.exp_capacity_rate);
        delete[] logits;
    }
    else if (test == "kv_cache") {
        uint32_t n_kv=2, hd=8, max=16;
        KVCache cache; cache.init(1, n_kv, hd, max);
        for (uint32_t p = 0; p < 4; p++)
            for (uint32_t h = 0; h < n_kv; h++)
                for (uint32_t d = 0; d < hd; d++) {
                    cache.key_caches[0][(h*max+p)*hd+d] = rand_float();
                    cache.value_caches[0][(h*max+p)*hd+d] = rand_float();
                }
        cache.current_len = 4;
        printf("KV Cache: entries=%d/%d key[0]=%.3f\n", cache.current_len, max, cache.key_caches[0][0]);
    }
    else if (test == "tokenizer") {
        HunyuanTokenizer tok;
        if (!tok.load("tokenizer.json")) { printf("FAILED to load tokenizer.json\n"); return 1; }
        auto ids = tok.encode("a lovely cat", true, true);
        printf("Tokenizer: 'a lovely cat' -> [");
        for (size_t i = 0; i < ids.size(); i++) printf("%d%s", ids[i], i+1<ids.size()?", ":"");
        printf("]\n  decoded: '%s'\n  vocab=%d\n", tok.decode(ids,true).c_str(), tok.vocab_size());
    }
    else if (test == "rope_2d") {
        uint32_t seq=8, hd=128, txt=4, ih=2, iw=2;
        float* cos = new float[seq*hd];
        float* sin = new float[seq*hd];
        build_2d_rope(seq, hd, txt, ih, iw, 10000.0f, cos, sin);
        printf("2D RoPE [seq=%d txt=%d img=%dx%d]: cos[0..3]=%.3f,%.3f,%.3f,%.3f\n",
               seq, txt, ih, iw, cos[0], cos[1], cos[2], cos[3]);
        // Verify energy preservation
        float* q = new float[seq*4*hd];
        for (size_t i = 0; i < seq*4*hd; i++) q[i] = rand_float();
        float rms_before = tensor_rms(q, seq*4*hd);
        float* k = new float[seq*2*hd];
        for (size_t i = 0; i < seq*2*hd; i++) k[i] = rand_float();
        apply_rope_2d_f32(q, k, cos, sin, seq, 4, 2, hd);
        float rms_after = tensor_rms(q, seq*4*hd);
        printf("  Q RMS before=%.6f after=%.6f (energy preserved: %s)\n",
               rms_before, rms_after, fabsf(rms_before-rms_after)<0.001 ? "YES" : "NO");
        delete[] cos; delete[] sin; delete[] q; delete[] k;
    }
    else {
        printf("Unknown test: %s\n", test.c_str());
        printf("Tests: groupnorm2d groupnorm3d conv2d conv3d layernorm softmax_topk moe_full kv_cache tokenizer rope_2d\n");
        return 1;
    }
    printf("PASS: %s\n", test.c_str());
    return 0;
}
