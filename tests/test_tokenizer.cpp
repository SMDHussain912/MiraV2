#include "core/action.hpp"
#include "core/intent.hpp"
#include "test_util.hpp"
#include "tokenizer/tokenizer.hpp"
#include <vector>

using mira::Intent;
using mira::Tokenizer;
using mira::TokenStatus;
using mira::TokenType;

void test_stripped()
{
    Tokenizer t;
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"open firefox", "firefox"},
        {"launch blender", "blender"},
        {"start vscode", "vscode"},
        {"run steam", "steam"},
    };
    for (const auto& c : cases)
    {
        auto r = t.process(c.first, Intent::OpenApplication);
        MIRA_CHECK_EQ(r.type, TokenType::TARGET);
        MIRA_CHECK_EQ(r.value, c.second);
        MIRA_CHECK_EQ(r.status, TokenStatus::Success);
        MIRA_CHECK_EQ(r.valid, true);
    }
}

void test_casing_and_spaces()
{
    Tokenizer t;
    auto r1 = t.process("OPEN Unity Editor", Intent::OpenApplication);
    MIRA_CHECK_EQ(r1.value, std::string("Unity Editor"));
    MIRA_CHECK_EQ(r1.status, TokenStatus::Success);
    MIRA_CHECK_EQ(r1.valid, true);

    auto r2 = t.process("   open firefox   ", Intent::OpenApplication);
    MIRA_CHECK_EQ(r2.value, std::string("firefox"));
    MIRA_CHECK_EQ(r2.status, TokenStatus::Success);

    auto r3 = t.process("open    firefox", Intent::OpenApplication);
    MIRA_CHECK_EQ(r3.value, std::string("firefox"));
    MIRA_CHECK_EQ(r3.status, TokenStatus::Success);
}

void test_punctuation_and_nouns()
{
    Tokenizer t;
    auto r1 = t.process("Open Firefox!", Intent::OpenApplication);
    MIRA_CHECK_EQ(r1.value, std::string("Firefox!"));
    MIRA_CHECK_EQ(r1.status, TokenStatus::Success);

    auto r2 = t.process("launch 7-Zip File Manager", Intent::OpenApplication);
    MIRA_CHECK_EQ(r2.value, std::string("7-Zip File Manager"));
    MIRA_CHECK_EQ(r2.status, TokenStatus::Success);

    auto r3 = t.process("open openoffice", Intent::OpenApplication);
    MIRA_CHECK_EQ(r3.value, std::string("openoffice"));
    MIRA_CHECK_EQ(r3.status, TokenStatus::Success);

    auto r4 = t.process("launch steam runtime", Intent::OpenApplication);
    MIRA_CHECK_EQ(r4.value, std::string("steam runtime"));
    MIRA_CHECK_EQ(r4.status, TokenStatus::Success);
}

void test_missing_target()
{
    Tokenizer t;
    for (const char* verb : {"open", "Open", "LAUNCH", "start", "run"})
    {
        auto r = t.process(verb, Intent::OpenApplication);
        MIRA_CHECK_EQ(r.type, TokenType::TARGET);
        MIRA_CHECK_EQ(r.value, std::string(""));
        MIRA_CHECK_EQ(r.status, TokenStatus::MissingTarget);
        MIRA_CHECK_EQ(r.valid, false);
    }

    auto r_spaces = t.process("open   ", Intent::OpenApplication);
    MIRA_CHECK_EQ(r_spaces.type, TokenType::TARGET);
    MIRA_CHECK_EQ(r_spaces.value, std::string(""));
    MIRA_CHECK_EQ(r_spaces.status, TokenStatus::MissingTarget);
    MIRA_CHECK_EQ(r_spaces.valid, false);
}

void test_empty_and_unsupported()
{
    Tokenizer t;
    auto r1 = t.process("", Intent::OpenApplication);
    MIRA_CHECK_EQ(r1.status, TokenStatus::EmptyInput);
    MIRA_CHECK_EQ(r1.valid, false);

    auto r2 = t.process("   ", Intent::OpenApplication);
    MIRA_CHECK_EQ(r2.status, TokenStatus::EmptyInput);
    MIRA_CHECK_EQ(r2.valid, false);

    const std::vector<Intent> unsupp = {
        Intent::SearchWeb,
        Intent::ReadScreen,
        Intent::TypeText,
        Intent::FileOperation,
        Intent::SystemControl,
        Intent::Conversation,
        Intent::Unknown,
    };
    for (Intent i : unsupp)
    {
        auto r = t.process("find something", i);
        MIRA_CHECK_EQ(r.type, TokenType::UNKNOWN);
        MIRA_CHECK_EQ(r.status, TokenStatus::UnsupportedIntent);
        MIRA_CHECK_EQ(r.valid, false);
    }
}

void test_to_string()
{
    MIRA_CHECK_EQ(mira::to_string(TokenStatus::Success), std::string("success"));
    MIRA_CHECK_EQ(mira::to_string(TokenStatus::EmptyInput), std::string("empty_input"));
    MIRA_CHECK_EQ(mira::to_string(TokenStatus::MissingTarget), std::string("missing_target"));
    MIRA_CHECK_EQ(mira::to_string(TokenStatus::UnsupportedIntent), std::string("unsupported_intent"));

    MIRA_CHECK_EQ(mira::to_string(TokenType::TARGET), std::string("target"));
    MIRA_CHECK_EQ(mira::to_string(TokenType::QUERY), std::string("query"));
    MIRA_CHECK_EQ(mira::to_string(TokenType::COMMAND), std::string("command"));
    MIRA_CHECK_EQ(mira::to_string(TokenType::TEXT), std::string("text"));
    MIRA_CHECK_EQ(mira::to_string(TokenType::PATH), std::string("path"));
    MIRA_CHECK_EQ(mira::to_string(TokenType::UNKNOWN), std::string("unknown"));
}

int main()
{
    test_stripped();
    test_casing_and_spaces();
    test_punctuation_and_nouns();
    test_missing_target();
    test_empty_and_unsupported();
    test_to_string();

    return mira_test::finish("tokenizer");
}
