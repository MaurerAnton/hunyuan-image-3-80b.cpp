cmake_minimum_required(VERSION 3.12)
project(hunyuan.cpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ggml library path (use local submodule)
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml)

# Build ggml as a static library
add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)

# Main executable
add_executable(hunyuan_test
    src/main_test.cpp
    src/hunyuan_model.cpp
    src/safetensors_loader.cpp
    src/rms_norm.cpp
    src/attention.cpp
    src/decoder_layer.cpp
    src/hunyuan_full_model.cpp
    src/hunyuan_gguf_loader.cpp
    src/hunyuan_image_gen.cpp
    src/hunyuan_vae.cpp
    src/hunyuan_moe_full.cpp
    src/hunyuan_extended.cpp
    src/hunyuan_tokenizer.cpp
    src/hunyuan_vit.cpp
    src/hunyuan_patches.cpp
)

target_include_directories(hunyuan_test PRIVATE
    ${GGML_DIR}/include
    src/
)

target_link_libraries(hunyuan_test PRIVATE ggml)

# Python reference test script stays as script, not compiled
