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
    // Single-pass: scan for "vocab" and "merges" objects
    // Vocab: {"token": id, ...} — key is string, value is int
    // Merges: ["tok1 tok2", ...]
    
    size_t p = 0;
    
    // Find "vocab":{...}
    p = j.find("\"vocab\"");
    if (p != std::string::npos) {
        p = j.find('{', p);
        if (p != std::string::npos) {
            p++; // enter {
            while (p < j.size()) {
                // Skip whitespace and commas
                while (p < j.size() && (j[p] == ' ' || j[p] == '\n' || j[p] == '\r' || j[p] == '\t' || j[p] == ',')) p++;
                if (p >= j.size() || j[p] == '}') break;
                
                // Read key string
                if (j[p] != '"') break;
                p++; // skip opening quote
                std::string token_text;
                while (p < j.size() && j[p] != '"') {
                    if (j[p] == '\\') { p++; if (p < j.size()) token_text += j[p]; }
                    else token_text += j[p];
                    p++;
                }
                if (p < j.size()) p++; // skip closing quote
                
                // Skip whitespace and colon
                while (p < j.size() && (j[p] == ' ' || j[p] == '\n' || j[p] == '\t' || j[p] == ':')) p++;
                
                // Read value (integer)
                int32_t id = 0;
                while (p < j.size() && j[p] >= '0' && j[p] <= '9') {
                    id = id * 10 + (j[p] - '0');
                    p++;
                }
                
                if (!token_text.empty() && id >= 0) {
                    if ((size_t)id >= vocab_.size()) vocab_.resize(id + 1);
                    vocab_[id].id = id;
                    vocab_[id].text = token_text;
                    vocab_[id].type = 1;
                    token_to_id_[token_text] = id;
                }
            }
        }
    }
    
    // Find "merges":[...]
    p = j.find("\"merges\"");
    if (p != std::string::npos) {
        p = j.find('[', p);
        if (p != std::string::npos) {
            p++; // enter [
            int32_t rank = 0;
            while (p < j.size()) {
                while (p < j.size() && (j[p] == ' ' || j[p] == '\n' || j[p] == '\r' || j[p] == '\t' || j[p] == ',')) p++;
                if (p >= j.size() || j[p] == ']') break;
                
                // Expect inner array: ["tok1", "tok2"]
                if (j[p] == '[') {
                    p++; // skip [
                    
                    // Read first string
                    while (p < j.size() && j[p] != '"') p++;
                    if (p >= j.size()) break;
                    p++;
                    std::string tok1;
                    while (p < j.size() && j[p] != '"') {
                        if (j[p] == '\\') { p++; if (p < j.size()) tok1 += j[p]; }
                        else tok1 += j[p];
                        p++;
                    }
                    if (p < j.size()) p++;
                    
                    // Skip comma
                    while (p < j.size() && j[p] != '"' && j[p] != ']') p++;
                    
                    // Read second string
                    if (p < j.size() && j[p] == '"') {
                        p++;
                        std::string tok2;
                        while (p < j.size() && j[p] != '"') {
                            if (j[p] == '\\') { p++; if (p < j.size()) tok2 += j[p]; }
                            else tok2 += j[p];
                            p++;
                        }
                        if (p < j.size()) p++;
                        
                        if (!tok1.empty() && !tok2.empty()) {
                            merges_.push_back({tok1, tok2});
                            merge_ranks_[tok1 + " " + tok2] = rank++;
                        }
                    }
                    
                    // Skip to end of inner array
                    while (p < j.size() && j[p] != ']') p++;
                    if (p < j.size()) p++; // skip ]
                } else {
                    break;
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
    
    // GPT-2 regex equivalent: split on patterns
    // Simplified: split on whitespace, preserving spaces as prefix
    size_t i = 0;
    while (i < text.size()) {
        // Collect leading whitespace
        size_t space_start = i;
        while (i < text.size() && isspace(text[i])) i++;
        std::string spaces = text.substr(space_start, i - space_start);
        
        if (i >= text.size()) {
            // Trailing whitespace
            if (!spaces.empty()) words.push_back(spaces);
            break;
        }
        
        // Collect word (letters/digits/punctuation)
        size_t word_start = i;
        if (isalpha(text[i]) || text[i] >= 0x80) {
            while (i < text.size() && (isalpha(text[i]) || text[i] >= 0x80)) i++;
        } else if (isdigit(text[i])) {
            while (i < text.size() && isdigit(text[i])) i++;
        } else {
            // Single punctuation
            i++;
        }
        std::string word = text.substr(word_start, i - word_start);
        
        // GPT-2 style: space is prefixed to the word using 'Ġ' (U+0120)
        // 'Ġ' represents a space in GPT-2 tokenizer convention
        if (!spaces.empty()) {
            // Last space becomes 'Ġ' prefix, preceding spaces become separate tokens
            if (spaces.size() > 1) {
                for (size_t s = 0; s < spaces.size() - 1; s++)
                    words.push_back("Ġ");
            }
            words.push_back("Ġ" + word);
        } else {
            words.push_back(word);
        }
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
            std::string t = vocab_[id].text;
            // Convert 'Ġ' back to space
            for (size_t c = 0; c < t.size(); c++) {
                if ((unsigned char)t[c] == 0xC4 && c+1 < t.size() && (unsigned char)t[c+1] == 0xA0) {
                    result += ' ';
                    c++; // skip second byte of UTF-8 Ġ
                } else {
                    result += t[c];
                }
            }
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
