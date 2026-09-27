// Unit tests for the C++ WordPiece tokenizer (tamev/bert_wordpiece_tokenizer.*).
//
// The vocabulary is loaded from models/tamev-base/tokenizer.json so these tests
// exercise the real 30522-entry BERT vocabulary used at runtime.

#include "tamev/bert_wordpiece_tokenizer.hpp"
#include "core/paths.hpp"
#include "test_util.hpp"

#include <string>
#include <vector>

namespace
{

using mira::BertWordPieceTokenizer;
using mira::EncodedTokens;

std::string tokenizer_path()
{
    return mira::paths::resolve_data_path("models/tamev-base/tokenizer.json");
}

void loads_real_vocabulary()
{
    BertWordPieceTokenizer tok;
    MIRA_CHECK(!tok.is_loaded());
    mira::Result r = tok.load(tokenizer_path());
    MIRA_CHECK(r.is_ok());
    MIRA_CHECK(tok.is_loaded());
    MIRA_CHECK(tok.vocab_size() == 30522);
}

void cls_sep_padding_and_masks()
{
    BertWordPieceTokenizer tok;
    tok.load(tokenizer_path());

    EncodedTokens enc = tok.encode("hello", 8);
    MIRA_CHECK_EQ(enc.input_ids.size(), static_cast<std::size_t>(8));
    MIRA_CHECK_EQ(enc.attention_mask.size(), static_cast<std::size_t>(8));
    // [CLS]=101 ... [SEP]=102, then [PAD]=0 padding with zero mask.
    MIRA_CHECK_EQ(enc.input_ids[0], static_cast<int64_t>(101));
    MIRA_CHECK_EQ(enc.attention_mask[0], static_cast<int64_t>(1));
    MIRA_CHECK_EQ(enc.input_ids[2], static_cast<int64_t>(102));
    MIRA_CHECK_EQ(enc.attention_mask[2], static_cast<int64_t>(1));
    MIRA_CHECK_EQ(enc.input_ids[3], static_cast<int64_t>(0));
    MIRA_CHECK_EQ(enc.attention_mask[3], static_cast<int64_t>(0));
}

void lowercase_and_word_split()
{
    BertWordPieceTokenizer tok;
    tok.load(tokenizer_path());

    EncodedTokens lower = tok.encode("Hello", 32);
    EncodedTokens upper = tok.encode("HELLO", 32);
    MIRA_CHECK(lower.input_ids == upper.input_ids);

    EncodedTokens two = tok.encode("hello world", 32);
    // [CLS] hello world [SEP] => positions 1 and 2 are content tokens.
    MIRA_CHECK(two.input_ids[1] != two.input_ids[2]);
    MIRA_CHECK(two.input_ids[1] != static_cast<int64_t>(100)); // not [UNK]
    MIRA_CHECK(two.input_ids[2] != static_cast<int64_t>(100));
}

void subword_splitting_uses_continuation()
{
    BertWordPieceTokenizer tok;
    tok.load(tokenizer_path());

    // "playing" exists as a whole-word entry (id 2652) so greedy longest-match
    // keeps it whole; an unknown compound forces continuation pieces.
    MIRA_CHECK_EQ(tok.token_to_id("playing"), static_cast<int64_t>(2652));
    MIRA_CHECK_EQ(tok.token_to_id("##ing"), static_cast<int64_t>(2075));

    EncodedTokens whole = tok.encode("playing", 32);
    MIRA_CHECK_EQ(whole.input_ids[0], static_cast<int64_t>(101));
    MIRA_CHECK_EQ(whole.input_ids[1], static_cast<int64_t>(2652));
    MIRA_CHECK_EQ(whole.input_ids[2], static_cast<int64_t>(102));

    // "open_application" is not in the vocab: it must split into word pieces
    // ("open" + "application" or subword continuation) rather than [UNK].
    EncodedTokens split = tok.encode("open_application", 32);
    MIRA_CHECK(split.input_ids[1] == tok.token_to_id("open"));
    MIRA_CHECK(split.input_ids[1] != static_cast<int64_t>(100));
    MIRA_CHECK(split.input_ids[2] != static_cast<int64_t>(100));
}

void truncation_keeps_cls_and_drops_sep_when_full()
{
    BertWordPieceTokenizer tok;
    tok.load(tokenizer_path());

    // max_length=3 with 5 content tokens: [CLS] + first 1 content + no room for [SEP].
    EncodedTokens enc = tok.encode("hello world foo bar baz", 3);
    MIRA_CHECK_EQ(enc.input_ids.size(), static_cast<std::size_t>(3));
    MIRA_CHECK_EQ(enc.input_ids[0], static_cast<int64_t>(101));
    MIRA_CHECK_EQ(enc.attention_mask[0], static_cast<int64_t>(1));
    MIRA_CHECK_EQ(enc.attention_mask[2], static_cast<int64_t>(1));
}

void missing_file_is_rejected_not_crash()
{
    BertWordPieceTokenizer tok;
    mira::Result r = tok.load("/nonexistent/tokenizer.json");
    MIRA_CHECK(r.is_error());
    MIRA_CHECK(!tok.is_loaded());
}

} // namespace

int main()
{
    loads_real_vocabulary();
    cls_sep_padding_and_masks();
    lowercase_and_word_split();
    subword_splitting_uses_continuation();
    truncation_keeps_cls_and_drops_sep_when_full();
    missing_file_is_rejected_not_crash();
    return mira_test::finish("tamev_tokenizer");
}
