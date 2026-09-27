// Unit tests for the shared text helpers (core/text_utils.*).
//
// These helpers replaced two private copies of the same logic (one in cmdmgr,
// one in the tokenizer), so their behaviour is pinned here as well as through
// tests/test_cmdmgr_baseline.cpp.

#include "core/text_utils.hpp"
#include "test_util.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace
{

void normalize_lowercases_and_replaces_punctuation()
{
    MIRA_CHECK_EQ(
        mira::normalize_for_matching("Open Firefox!"),
        std::string("open firefox ")
    );

    MIRA_CHECK_EQ(
        mira::normalize_for_matching("Fire-fox"),
        std::string("fire fox")
    );

    MIRA_CHECK_EQ(
        mira::normalize_for_matching("What is on my screen?"),
        std::string("what is on my screen ")
    );

    MIRA_CHECK_EQ(mira::normalize_for_matching("MIRA"), std::string("mira"));
    MIRA_CHECK_EQ(mira::normalize_for_matching("12345"), std::string("12345"));
    MIRA_CHECK_EQ(mira::normalize_for_matching(""), std::string(""));
}

void trim_removes_only_surrounding_whitespace()
{
    MIRA_CHECK_EQ(mira::trim("  open firefox  "), std::string("open firefox"));
    MIRA_CHECK_EQ(mira::trim("open firefox"), std::string("open firefox"));
    MIRA_CHECK_EQ(mira::trim("open  firefox"), std::string("open  firefox"));
    MIRA_CHECK_EQ(mira::trim("\t open \n"), std::string("open"));
    MIRA_CHECK_EQ(mira::trim("   "), std::string(""));
    MIRA_CHECK_EQ(mira::trim(""), std::string(""));
}

void split_whitespace_collapses_runs()
{
    const std::vector<std::string> tokens =
        mira::split_whitespace("  open    firefox  ");

    MIRA_CHECK_EQ(tokens.size(), std::size_t{2});
    MIRA_CHECK_EQ(tokens[0], std::string("open"));
    MIRA_CHECK_EQ(tokens[1], std::string("firefox"));

    MIRA_CHECK_EQ(mira::split_whitespace("").size(), std::size_t{0});
    MIRA_CHECK_EQ(mira::split_whitespace("   ").size(), std::size_t{0});
}

void case_insensitive_equality()
{
    MIRA_CHECK(mira::equals_case_insensitive("Open", "open"));
    MIRA_CHECK(mira::equals_case_insensitive("OPEN", "open"));
    MIRA_CHECK(mira::equals_case_insensitive("", ""));
    MIRA_CHECK(!mira::equals_case_insensitive("open", "opens"));
    MIRA_CHECK(!mira::equals_case_insensitive("", "a"));
    MIRA_CHECK(!mira::equals_case_insensitive("open", "opem"));
}

void case_insensitive_prefix_stripping()
{
    std::string matched = "Open Firefox";
    MIRA_CHECK(mira::strip_prefix_case_insensitive(matched, "open "));
    MIRA_CHECK_EQ(matched, std::string("Firefox"));

    std::string unmatched = "Firefox";
    MIRA_CHECK(!mira::strip_prefix_case_insensitive(unmatched, "open "));
    MIRA_CHECK_EQ(unmatched, std::string("Firefox"));

    // The verb includes its trailing space, so a bare verb is not stripped.
    std::string bare_verb = "run";
    MIRA_CHECK(!mira::strip_prefix_case_insensitive(bare_verb, "run "));
    MIRA_CHECK_EQ(bare_verb, std::string("run"));
}

} // namespace

int main()
{
    normalize_lowercases_and_replaces_punctuation();
    trim_removes_only_surrounding_whitespace();
    split_whitespace_collapses_runs();
    case_insensitive_equality();
    case_insensitive_prefix_stripping();

    return mira_test::finish("text_utils");
}
