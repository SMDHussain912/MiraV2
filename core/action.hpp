#pragma once

#include <string>

#include "core/intent.hpp"

namespace mira
{

// What the tokenizer extracted from an utterance.
//
// The tokenizer runs *after* TAMEV has decided the intent, and returns the value
// that intent needs: an application reference, a search query, text to type, a
// path, or a system-control command.
enum class TokenType
{
    TARGET,
    QUERY,
    COMMAND,
    TEXT,
    PATH,
    UNKNOWN
};

struct TokenResult
{
    TokenType type = TokenType::UNKNOWN;
    std::string value;
    bool valid = false;
};

// A single unit of work: the decided intent, what was extracted for it, and the
// original transcript (for logging and for clarification prompts).
//
// This is the type that crosses from the decision/extraction stages into
// execution. Defined here rather than in the tokenizer so that the executor does
// not have to depend on a particular extraction implementation.
struct ActionRequest
{
    Intent intent = Intent::Unknown;
    TokenResult tokens;
    std::string original_text;
};

} // namespace mira
