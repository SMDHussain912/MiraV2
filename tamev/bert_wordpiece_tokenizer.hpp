#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/result.hpp"

namespace mira
{

// Output of tokenization, ready to be fed directly to ONNX model inputs.
struct EncodedTokens
{
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
};

// WordPiece tokenizer implementing BERT-style tokenization (ROADMAP Phase 6).
// Loads vocabulary directly from HuggingFace tokenizer.json (WordPiece vocab).
//
// Distinctly named BertWordPieceTokenizer to prevent naming collision with
// Mira's intent-aware Target Tokenizer (mira::Tokenizer).
class BertWordPieceTokenizer
{
public:
    BertWordPieceTokenizer() = default;
    ~BertWordPieceTokenizer() = default;

    // Loads vocabulary and special tokens from tokenizer.json.
    Result load(const std::string& tokenizer_json_path);

    // Checks if the tokenizer has a loaded vocabulary.
    bool is_loaded() const;

    // Encodes an input string, performing:
    //  1. Lowercasing and BERT punctuation / whitespace splitting
    //  2. WordPiece subword search (with "##" prefix)
    //  3. Insertion of [CLS] (101) and [SEP] (102)
    //  4. Truncation and [PAD] (0) padding to exactly max_length
    EncodedTokens encode(const std::string& text, size_t max_length = 128) const;

    // Returns the token ID for a given token string, or unk_token_id_ if not found.
    int64_t token_to_id(const std::string& token) const;

    // Vocab size
    size_t vocab_size() const;

private:
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    void tokenize_word(const std::string& word, std::vector<int64_t>& out_ids) const;

    std::unordered_map<std::string, int64_t> vocab_;
    int64_t pad_token_id_ = 0;
    int64_t unk_token_id_ = 100;
    int64_t cls_token_id_ = 101;
    int64_t sep_token_id_ = 102;
    int64_t mask_token_id_ = 103;
    bool is_loaded_ = false;
};

} // namespace mira
