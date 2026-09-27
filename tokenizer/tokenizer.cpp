#include "tokenizer/tokenizer.hpp"

#include "core/text_utils.hpp"

namespace mira
{

namespace
{

// Leading command verbs for the OpenApplication intent that carry no information
// about the target itself and are stripped before handoff to resolution.
const char* const kApplicationCommandVerbs[] = {
    "open",
    "launch",
    "start",
    "run"
};

} // namespace

TokenResult Tokenizer::process(const std::string& text, Intent intent)
{
    if (intent == Intent::OpenApplication)
    {
        return extract_application(text);
    }

    // Extraction for the other intents is defined in tokenizer/README.md and
    // implemented in ROADMAP Phase 8.
    //
    // Returning an invalid result with UnsupportedIntent is deliberate: downstream
    // stages (resolver, executor) must never act on an unextracted request.
    TokenResult result;
    result.type = TokenType::UNKNOWN;
    result.value.clear();
    result.status = TokenStatus::UnsupportedIntent;
    result.valid = false;

    return result;
}

TokenResult Tokenizer::extract_application(const std::string& text)
{
    TokenResult result;
    result.type = TokenType::TARGET;
    result.value.clear();
    result.valid = false;

    const std::string trimmed = trim(text);
    if (trimmed.empty())
    {
        result.status = TokenStatus::EmptyInput;
        return result;
    }

    std::string candidate = trimmed;

    for (const char* verb : kApplicationCommandVerbs)
    {
        // 1. Bare verb without target: e.g. "open", "Launch"
        if (equals_case_insensitive(candidate, verb))
        {
            result.status = TokenStatus::MissingTarget;
            return result;
        }

        // 2. Verb followed by space: e.g. "open ", "launch  "
        const std::string verb_prefix = std::string(verb) + " ";
        if (strip_prefix_case_insensitive(candidate, verb_prefix))
        {
            // Clean up any remaining leading whitespace between verb and target
            // while preserving interior spacing, target casing, and punctuation.
            candidate = trim(candidate);
            break;
        }
    }

    if (candidate.empty())
    {
        result.status = TokenStatus::MissingTarget;
        return result;
    }

    result.value = candidate;
    result.status = TokenStatus::Success;
    result.valid = true;

    return result;
}

} // namespace mira


