#include "tamev/bert_wordpiece_tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mira
{

namespace
{

bool is_bert_punctuation(char c)
{
    // BERT considers all ASCII punctuation characters as punctuation
    return (c >= 33 && c <= 47) ||
           (c >= 58 && c <= 64) ||
           (c >= 91 && c <= 96) ||
           (c >= 123 && c <= 126);
}

} // namespace

Result BertWordPieceTokenizer::load(const std::string& tokenizer_json_path)
{
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open())
    {
        return Result::rejected("cannot open tokenizer file: " + tokenizer_json_path);
    }

    json root;
    try
    {
        file >> root;
    }
    catch (const json::parse_error& e)
    {
        return Result::rejected("malformed tokenizer JSON", e.what());
    }

    if (!root.contains("model") || !root["model"].is_object() ||
        !root["model"].contains("vocab") || !root["model"]["vocab"].is_object())
    {
        return Result::rejected("missing model.vocab in tokenizer.json");
    }

    vocab_.clear();
    const auto& vocab_obj = root["model"]["vocab"];
    for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it)
    {
        if (it.value().is_number_integer())
        {
            vocab_[it.key()] = it.value().get<int64_t>();
        }
    }

    if (vocab_.count("[PAD]")) pad_token_id_ = vocab_["[PAD]"];
    if (vocab_.count("[UNK]")) unk_token_id_ = vocab_["[UNK]"];
    if (vocab_.count("[CLS]")) cls_token_id_ = vocab_["[CLS]"];
    if (vocab_.count("[SEP]")) sep_token_id_ = vocab_["[SEP]"];
    if (vocab_.count("[MASK]")) mask_token_id_ = vocab_["[MASK]"];

    is_loaded_ = !vocab_.empty();
    if (!is_loaded_)
    {
        return Result::rejected("vocabulary is empty");
    }

    return Result::ok();
}

bool BertWordPieceTokenizer::is_loaded() const
{
    return is_loaded_;
}

size_t BertWordPieceTokenizer::vocab_size() const
{
    return vocab_.size();
}

int64_t BertWordPieceTokenizer::token_to_id(const std::string& token) const
{
    auto it = vocab_.find(token);
    if (it != vocab_.end())
    {
        return it->second;
    }
    return unk_token_id_;
}

std::vector<std::string> BertWordPieceTokenizer::pre_tokenize(const std::string& text) const
{
    std::vector<std::string> words;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i)
    {
        unsigned char uc = static_cast<unsigned char>(text[i]);
        char c = static_cast<char>(std::tolower(uc));

        if (std::isspace(uc))
        {
            if (!current.empty())
            {
                words.push_back(current);
                current.clear();
            }
        }
        else if (is_bert_punctuation(c))
        {
            if (!current.empty())
            {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, c));
        }
        else
        {
            current += c;
        }
    }

    if (!current.empty())
    {
        words.push_back(current);
    }

    return words;
}

void BertWordPieceTokenizer::tokenize_word(const std::string& word, std::vector<int64_t>& out_ids) const
{
    if (word.empty())
    {
        return;
    }

    if (word.size() > 100) // max_input_chars_per_word
    {
        out_ids.push_back(unk_token_id_);
        return;
    }

    size_t start = 0;
    std::vector<int64_t> subword_ids;
    bool is_bad = false;

    while (start < word.size())
    {
        size_t end = word.size();
        std::string cur_substr;
        int64_t match_id = -1;

        while (start < end)
        {
            std::string sub = word.substr(start, end - start);
            if (start > 0)
            {
                sub = "##" + sub;
            }

            auto it = vocab_.find(sub);
            if (it != vocab_.end())
            {
                cur_substr = sub;
                match_id = it->second;
                break;
            }
            --end;
        }

        if (match_id == -1)
        {
            is_bad = true;
            break;
        }

        subword_ids.push_back(match_id);
        start = end;
    }

    if (is_bad)
    {
        out_ids.push_back(unk_token_id_);
    }
    else
    {
        out_ids.insert(out_ids.end(), subword_ids.begin(), subword_ids.end());
    }
}

EncodedTokens BertWordPieceTokenizer::encode(const std::string& text, size_t max_length) const
{
    EncodedTokens result;
    result.input_ids.reserve(max_length);
    result.attention_mask.reserve(max_length);

    if (max_length == 0)
    {
        return result;
    }

    std::vector<int64_t> content_ids;
    std::vector<std::string> words = pre_tokenize(text);
    for (const auto& w : words)
    {
        tokenize_word(w, content_ids);
    }

    // [CLS] + content + [SEP]
    result.input_ids.push_back(cls_token_id_);
    result.attention_mask.push_back(1);

    size_t max_content = (max_length >= 2) ? (max_length - 2) : 0;
    size_t take = std::min(content_ids.size(), max_content);

    for (size_t i = 0; i < take; ++i)
    {
        result.input_ids.push_back(content_ids[i]);
        result.attention_mask.push_back(1);
    }

    if (result.input_ids.size() < max_length)
    {
        result.input_ids.push_back(sep_token_id_);
        result.attention_mask.push_back(1);
    }

    while (result.input_ids.size() < max_length)
    {
        result.input_ids.push_back(pad_token_id_);
        result.attention_mask.push_back(0);
    }

    return result;
}

} // namespace mira

