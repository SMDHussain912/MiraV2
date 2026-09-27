// Tests for intent-aware extraction (tokenizer/).
//
// Phase 1 changed the tokenizer's public signature (an Intent instead of a free
// string) and moved the shared types into core/, but it must not change what the
// OpenApplication extraction does. The expectations below therefore describe the
// behaviour of the pre-Phase-1 implementation, including the two known warts
// marked "known limitation" - those are fixed deliberately in Phase 2, not by
// accident here.
//
// The tokenizer must never be callable without an intent: process() requires one.

#include "core/action.hpp"
#include "core/intent.hpp"
#include "test_util.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace
{

using mira::Intent;
using mira::Tokenizer;
using mira::TokenType;

void the_four_command_verbs_are_stripped()
{
    Tokenizer tokenizer;

    const std::vector<std::pair<std::string, std::string>> cases = {
        { "open firefox",   "firefox" },
        { "launch blender", "blender" },
        { "start vscode",   "vscode"  },
        { "run steam",      "steam"   },
    };

    for (const auto& test_case : cases)
    {
        const mira::TokenResult result =
            tokenizer.process(test_case.first, Intent::OpenApplication);

        MIRA_CHECK_EQ(result.type, TokenType::TARGET);
        MIRA_CHECK_EQ(result.value, test_case.second);
        MIRA_CHECK_EQ(result.valid, true);
    }
}

void verb_matching_ignores_case_and_the_target_keeps_its_case()
{
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("OPEN Unity Editor", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.type, TokenType::TARGET);
    MIRA_CHECK_EQ(result.value, std::string("Unity Editor"));
    MIRA_CHECK_EQ(result.valid, true);
}

void surrounding_whitespace_is_trimmed()
{
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("   open firefox   ", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.value, std::string("firefox"));
    MIRA_CHECK_EQ(result.valid, true);
}

void multi_word_targets_are_returned_whole()
{
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("open unity editor", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.value, std::string("unity editor"));
    MIRA_CHECK_EQ(result.valid, true);
}

void punctuation_inside_the_target_is_preserved()
{
    // Unlike the legacy command splitter, the tokenizer does not normalise the
    // extracted value: "Firefox!" stays "Firefox!". Resolving it is the
    // resolver's job and happens later (ROADMAP Phase 3).
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("Open Firefox!", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.value, std::string("Firefox!"));
    MIRA_CHECK_EQ(result.valid, true);
}

void whitespace_after_the_verb_is_preserved()
{
    // Known limitation (kept unchanged in Phase 1, fixed in Phase 2): the verb
    // includes a single trailing space, so extra spaces remain in the value.
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("open   firefox", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.value, std::string("  firefox"));
    MIRA_CHECK_EQ(result.valid, true);
}

void a_verb_only_utterance_is_currently_accepted()
{
    // Known limitation (kept unchanged in Phase 1, fixed in Phase 2): a bare
    // verb does not match a verb+space prefix, so it is returned as a target.
    Tokenizer tokenizer;

    const mira::TokenResult result =
        tokenizer.process("Open", Intent::OpenApplication);

    MIRA_CHECK_EQ(result.value, std::string("Open"));
    MIRA_CHECK_EQ(result.valid, true);
}

void empty_input_is_invalid()
{
    Tokenizer tokenizer;

    const mira::TokenResult empty =
        tokenizer.process("", Intent::OpenApplication);

    MIRA_CHECK_EQ(empty.type, TokenType::TARGET);
    MIRA_CHECK_EQ(empty.value, std::string(""));
    MIRA_CHECK_EQ(empty.valid, false);

    const mira::TokenResult blank =
        tokenizer.process("     ", Intent::OpenApplication);

    MIRA_CHECK_EQ(blank.value, std::string(""));
    MIRA_CHECK_EQ(blank.valid, false);
}

void intents_without_extraction_return_an_invalid_result()
{
    // Extraction is implemented for OpenApplication only (ROADMAP Phase 8 adds
    // the others one at a time). An unimplemented intent must not look like a
    // successful extraction, because a caller would act on the empty value.
    Tokenizer tokenizer;

    const std::vector<Intent> not_implemented = {
        Intent::SearchWeb,
        Intent::ReadScreen,
        Intent::TypeText,
        Intent::FileOperation,
        Intent::SystemControl,
        Intent::Conversation,
        Intent::Unknown,
    };

    for (Intent intent : not_implemented)
    {
        const mira::TokenResult result =
            tokenizer.process("search the web for python tutorials", intent);

        MIRA_CHECK_EQ(result.type, TokenType::UNKNOWN);
        MIRA_CHECK_EQ(result.value, std::string(""));
        MIRA_CHECK_EQ(result.valid, false);
    }
}

} // namespace

int main()
{
    the_four_command_verbs_are_stripped();
    verb_matching_ignores_case_and_the_target_keeps_its_case();
    surrounding_whitespace_is_trimmed();
    multi_word_targets_are_returned_whole();
    punctuation_inside_the_target_is_preserved();
    whitespace_after_the_verb_is_preserved();
    a_verb_only_utterance_is_currently_accepted();
    empty_input_is_invalid();
    intents_without_extraction_return_an_invalid_result();

    return mira_test::finish("tokenizer");
}
