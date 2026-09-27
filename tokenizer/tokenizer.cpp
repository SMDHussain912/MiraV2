#include "tokenizer/tokenizer.hpp"

#include "core/text_utils.hpp"

namespace mira
{

namespace
{

// Leading command verbs that carry no information for the OpenApplication
// intent and are removed before the application reference is returned.
const char* const kApplicationCommandVerbs[] = {
    "open ",
    "launch ",
    "start ",
    "run "
};

} // namespace

TokenResult Tokenizer::process(const std::string& text, Intent intent)
{
    if (intent == Intent::OpenApplication)
    {
        return extract_application(text);
    }

    // Extraction for this intent is not implemented yet (ROADMAP Phase 8).
    // Returning an invalid result is deliberate: callers must not act on a value
    // the tokenizer did not actually extract.
    TokenResult result;
    result.type = TokenType::UNKNOWN;
    result.value.clear();
    result.valid = false;

    return result;
}

TokenResult Tokenizer::extract_application(const std::string& text)
{
    TokenResult result;

    result.type = TokenType::TARGET;
    result.value = trim(text);
    result.valid = false;

    for (const char* verb : kApplicationCommandVerbs)
    {
        if (strip_prefix_case_insensitive(result.value, verb))
        {
            break;
        }
    }

    result.valid = !result.value.empty();

    return result;
}

} // namespace mira

