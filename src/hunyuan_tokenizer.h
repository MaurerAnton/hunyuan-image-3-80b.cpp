#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

// ============================================================
// Minimal BPE Tokenizer for Hunyuan Image 3
//
// Loads tokenizer.json (HuggingFace format) and provides
// encode/decode matching PreTrainedTokenizerFast behavior.
//
// Vocabulary: 133,120 tokens
// Special tokens:
//   BOS: 127958 (<|startoftext|>)
//   EOS: 127957 (<|endoftext|>)
//   PAD: 128009 (<pad>)
//   IMG: 128006 (image token)
// ============================================================

struct BPEToken {
    std::string text;
    float score = 0.0f;
    int32_t id = 0;
    int32_t type = 1;  // 1=normal, 2=control, 3=unused, 4=user_defined, etc.
};

class HunyuanTokenizer {
public:
    HunyuanTokenizer() = default;
    
    // Load from tokenizer.json file
    bool load(const std::string& path);
    
    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true, bool add_eos = true) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;
    
    // Single token lookup
    std::string id_to_token(int32_t id) const;
    int32_t token_to_id(const std::string& token) const;
    
    // Special token IDs
    int32_t bos_id() const { return bos_token_id_; }
    int32_t eos_id() const { return eos_token_id_; }
    int32_t pad_id() const { return pad_token_id_; }
    int32_t image_token_id() const { return image_token_id_; }
    int32_t vocab_size() const { return (int32_t)vocab_.size(); }
    
private:
    // Vocabulary: id → token info
    std::vector<BPEToken> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    
    // BPE merge rules: (token1, token2) → merged_token
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> merge_ranks_;  // "tok1 tok2" → rank
    
    // Special token IDs
    int32_t bos_token_id_ = 127958;
    int32_t eos_token_id_ = 127957;
    int32_t pad_token_id_ = 128009;
    int32_t image_token_id_ = 128006;
    int32_t text_start_id_ = 6;
    int32_t text_end_id_ = 7;
    
    // Regex for pre-tokenization (GPT-2 style)
    // Matches: contractions, letters, digits, punctuation, whitespace
    std::string pretokenize_pattern_;
    
    // BPE encode a single pretokenized word
    std::vector<int32_t> bpe_encode_word(const std::string& word) const;
    
    // Parse tokenizer.json (minimal JSON parser)
    bool parse_tokenizer_json(const std::string& json_str);
    
    // Extract string value from JSON
    static std::string json_extract_string(const std::string& json, size_t& pos);
    static int32_t json_extract_int(const std::string& json, size_t& pos);
    static float json_extract_float(const std::string& json, size_t& pos);
    static void json_skip_value(const std::string& json, size_t& pos);
    static void json_skip_whitespace(const std::string& json, size_t& pos);
    
    // GPT-2 style pre-tokenization regex matching
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<std::string> pretokenize(const std::string& text) const;
};
