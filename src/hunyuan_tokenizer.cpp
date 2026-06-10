#include "hunyuan_tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <sstream>
#include <regex>

// ============================================================
// JSON helpers
// ============================================================

void HunyuanTokenizer::json_skip_whitespace(const std::string& j, size_t& p) {
    while (p < j.size() && (j[p] == ' ' || j[p] == '\n' || j[p] == '\r' || j[p] == '\t')) p++;
}

void HunyuanTokenizer::json_skip_value(const std::string& j, size_t& p) {
    json_skip_whitespace(j, p);
    if (p >= j.size()) return;
    
    if (j[p] == '"') { p++; while (p < j.size() && j[p] != '"') { if (j[p] == '\\') p++; p++; } p++; }
    else if (j[p] == '{') { int d = 1; p++; while (p < j.size() && d > 0) { if (j[p] == '{') d++; if (j[p] == '}') d--; p++; } }
    else if (j[p] == '[') { int d = 1; p++; while (p < j.size() && d > 0) { if (j[p] == '[') d++; if (j[p] == ']') d--; p++; } }
    else { while (p < j.size() && j[p] != ',' && j[p] != '}' && j[p] != ']') p++; }
}

std::string HunyuanTokenizer::json_extract_string(const std::string& j, size_t& p) {
    json_skip_whitespace(j, p);
    if (j[p] != '"') return "";
    p++;
    std::string result;
    while (p < j.size() && j[p] != '"') {
        if (j[p] == '\\') { p++; if (p < j.size()) result += j[p]; }
        else result += j[p];
        p++;
    }
    if (p < j.size()) p++;
    return result;
}

int32_t HunyuanTokenizer::json_extract_int(const std::string& j, size_t& p) {
    json_skip_whitespace(j, p);
    std::string num;
    while (p < j.size() && (isdigit(j[p]) || j[p] == '-')) { num += j[p]; p++; }
    return num.empty() ? 0 : atoi(num.c_str());
}

float HunyuanTokenizer::json_extract_float(const std::string& j, size_t& p) {
    json_skip_whitespace(j, p);
    std::string num;
    while (p < j.size() && (isdigit(j[p]) || j[p] == '-' || j[p] == '.' || j[p] == 'e' || j[p] == 'E' || j[p] == '+')) { num += j[p]; p++; }
    return num.empty() ? 0.0f : (float)atof(num.c_str());
}

// ============================================================
// Parse tokenizer.json
// ============================================================

bool HunyuanTokenizer::parse_tokenizer_json(const std::string& j) {
    size_t p = 0;
    
    // Find "added_tokens" array
    p = j.find("\"added_tokens\"");
    if (p != std::string::npos) {
        p = j.find('[', p);
        p++;
        while (p < j.size() && j[p] != ']') {
            if (j[p] == '{') {
                BPEToken tok;
                p++; // skip {
                while (p < j.size() && j[p] != '}') {
                    std::string key = json_extract_string(j, p);
                    json_skip_whitespace(j, p);
                    if (p < j.size() && j[p] == ':') p++;
                    if (key == "id") tok.id = json_extract_int(j, p);
                    else if (key == "content") tok.text = json_extract_string(j, p);
                    else if (key == "special") { bool s = (json_extract_string(j, p) == "true" || j[p-4] == 't'); tok.type = s ? 2 : 1; }
                    else json_skip_value(j, p);
                    json_skip_whitespace(j, p);
                    if (p < j.size() && j[p] == ',') p++;
                }
                p++; // skip }
                // Store
                if (tok.id >= 0 && !tok.text.empty()) {
                    if ((size_t)tok.id >= vocab_.size()) vocab_.resize(tok.id + 1);
                    vocab_[tok.id] = tok;
                    token_to_id_[tok.text] = tok.id;
                }
            }
            json_skip_whitespace(j, p);
            if (p < j.size() && j[p] == ',') p++;
        }
    }
    
    // Find "model" → "vocab" object
    p = j.find("\"model\"");
    if (p == std::string::npos) return !vocab_.empty();
    p = j.find('{', p); p++;
    
    // "vocab" is inside "model"
    p = j.find("\"vocab\"", p);
    if (p != std::string::npos) {
        p = j.find('{', p); p++;
        while (p < j.size() && j[p] != '}') {
            std::string token_text = json_extract_string(j, p);
            json_skip_whitespace(j, p);
            if (p < j.size() && j[p] == ':') p++;
            int32_t id = json_extract_int(j, p);
            json_skip_whitespace(j, p);
            if (p < j.size() && j[p] == ',') p++;
            
            if (id >= 0 && !token_text.empty()) {
                if ((size_t)id >= vocab_.size()) vocab_.resize(id + 1);
                if (vocab_[id].text.empty()) {
                    vocab_[id].id = id;
                    vocab_[id].text = token_text;
                    vocab_[id].type = 1;
                    token_to_id_[token_text] = id;
                }
            }
        }
    }
    
    // Find "merges" array
    p = j.find("\"merges\"");
    if (p != std::string::npos) {
        p = j.find('[', p); p++;
        int32_t rank = 0;
        while (p < j.size() && j[p] != ']') {
            std::string merge_str = json_extract_string(j, p);
            json_skip_whitespace(j, p);
            if (p < j.size() && j[p] == ',') p++;
            
            if (!merge_str.empty()) {
                // Merge format: "tok1 tok2"
                size_t space = merge_str.find(' ');
                if (space != std::string::npos) {
                    std::string t1 = merge_str.substr(0, space);
                    std::string t2 = merge_str.substr(space + 1);
                    merges_.push_back({t1, t2});
                    merge_ranks_[merge_str] = rank++;
                }
            }
        }
    }
    
    printf("Tokenizer: %zu tokens, %zu merges\n", vocab_.size(), merges_.size());
    return !vocab_.empty();
}

// ============================================================
// Load from file
// ============================================================

bool HunyuanTokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open tokenizer: %s\n", path.c_str()); return false; }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_tokenizer_json(json);
}

// ============================================================
// Pre-tokenization (GPT-2 style regex)
// ============================================================

std::vector<std::string> HunyuanTokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> words;
    
    // GPT-2 pattern: contractions, letters, numbers, punctuation, whitespace
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    
    // Simplified implementation using character-by-character scanning
    size_t i = 0;
    while (i < text.size()) {
        // Skip whitespace at start of word, handle separately
        if (isspace(text[i])) {
            size_t start = i;
            while (i < text.size() && isspace(text[i])) i++;
            words.push_back(text.substr(start, i - start));
            continue;
        }
        
        // Check for contractions
        if (i + 2 < text.size() && text[i] == '\'' && 
            ((text[i+1] == 's' && (i+2 >= text.size() || !isalpha(text[i+2]))) ||
             (text[i+1] == 't' && (i+2 >= text.size() || !isalpha(text[i+2]))) ||
             (text[i+1] == 'm' && (i+2 >= text.size() || !isalpha(text[i+2]))) ||
             (text[i+1] == 'd' && (i+2 >= text.size() || !isalpha(text[i+2]))))) {
            words.push_back(text.substr(i, 2));
            i += 2;
            continue;
        }
        if (i + 3 < text.size() && text[i] == '\'' &&
            ((text.substr(i, 3) == "'re") || (text.substr(i, 3) == "'ve") || (text.substr(i, 3) == "'ll")) &&
            (i+3 >= text.size() || !isalpha(text[i+3]))) {
            words.push_back(text.substr(i, 3));
            i += 3;
            continue;
        }
        
        // Letters
        if (isalpha(text[i])) {
            size_t start = i;
            while (i < text.size() && isalpha(text[i])) i++;
            words.push_back(text.substr(start, i - start));
            continue;
        }
        
        // Digits
        if (isdigit(text[i])) {
            size_t start = i;
            while (i < text.size() && isdigit(text[i])) i++;
            words.push_back(text.substr(start, i - start));
            continue;
        }
        
        // Single punctuation/other
        words.push_back(text.substr(i, 1));
        i++;
    }
    
    return words;
}

// ============================================================
// BPE encode a single word
// ============================================================

std::vector<int32_t> HunyuanTokenizer::bpe_encode_word(const std::string& word) const {
    // Start with UTF-8 character-level tokens (or byte-level for GPT-2)
    // For GPT-2 BPE: convert each character to its byte representation + token lookup
    
    // Simplified: use character-level tokens initially
    std::vector<std::string> symbols;
    for (size_t i = 0; i < word.size(); ) {
        // Handle multi-byte UTF-8
        int len = 1;
        if ((word[i] & 0x80) == 0) len = 1;
        else if ((word[i] & 0xE0) == 0xC0) len = 2;
        else if ((word[i] & 0xF0) == 0xE0) len = 3;
        else if ((word[i] & 0xF8) == 0xF0) len = 4;
        
        std::string ch = word.substr(i, len);
        symbols.push_back(ch);
        i += len;
    }
    
    if (symbols.size() <= 1) {
        // No merges possible
        std::vector<int32_t> ids;
        for (auto& s : symbols) {
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) ids.push_back(it->second);
        }
        return ids;
    }
    
    // BPE merge loop
    while (symbols.size() > 1) {
        // Find the best merge (lowest rank = earliest merge)
        int32_t best_rank = INT32_MAX;
        size_t best_i = 0;
        
        for (size_t i = 0; i < symbols.size() - 1; i++) {
            std::string pair = symbols[i] + " " + symbols[i+1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        
        if (best_rank == INT32_MAX) break;  // No more merges
        
        // Apply merge
        std::string merged = symbols[best_i] + symbols[best_i+1];
        symbols[best_i] = merged;
        symbols.erase(symbols.begin() + best_i + 1);
    }
    
    // Convert final symbols to token IDs
    std::vector<int32_t> ids;
    for (auto& s : symbols) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) ids.push_back(it->second);
        else {
            // Fall back to UTF-8 byte encoding
            for (size_t b = 0; b < s.size(); b++) {
                std::string byte_str = std::string(1, s[b]);
                auto it2 = token_to_id_.find(byte_str);
                if (it2 != token_to_id_.end()) ids.push_back(it2->second);
            }
        }
    }
    return ids;
}

// ============================================================
// Encode text → token IDs
// ============================================================

std::vector<int32_t> HunyuanTokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int32_t> ids;
    
    if (add_bos) ids.push_back(bos_token_id_);
    
    // Pre-tokenize
    auto words = pretokenize(text);
    for (auto& word : words) {
        auto word_ids = bpe_encode_word(word);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }
    
    if (add_eos) ids.push_back(eos_token_id_);
    
    return ids;
}

// ============================================================
// Decode token IDs → text
// ============================================================

std::string HunyuanTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string result;
    for (auto id : ids) {
        if (skip_special && (id == bos_token_id_ || id == eos_token_id_ || id == pad_token_id_))
            continue;
        if (id >= 0 && (size_t)id < vocab_.size() && !vocab_[id].text.empty()) {
            result += vocab_[id].text;
        }
    }
    return result;
}

std::string HunyuanTokenizer::id_to_token(int32_t id) const {
    if (id >= 0 && (size_t)id < vocab_.size()) return vocab_[id].text;
    return "";
}

int32_t HunyuanTokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}
