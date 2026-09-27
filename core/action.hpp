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

std::string to_string(TokenType type);

// Canonical status states for token extraction failure and success.
//
// Kept small, typed, and consistent across all intent extractors to eliminate
// arbitrary error strings in the extraction stage.
enum class TokenStatus
{
    Success,
    EmptyInput,
    MissingTarget,
    UnsupportedIntent
};

std::string to_string(TokenStatus status);

struct TokenResult
{
    TokenType type = TokenType::UNKNOWN;
    std::string value;
    TokenStatus status = TokenStatus::EmptyInput;
    bool valid = false;
};

// A single unit of work: the decided intent, what was extracted for it, and the
// original transcript (for logging and for clarification prompts).
//
// This is the handoff type crossing from classification/extraction into
// resolution. Defined here rather than in the tokenizer so that resolvers
// and executors do not have to depend on a specific tokenizer implementation.
struct ActionRequest
{
    Intent intent = Intent::Unknown;
    TokenResult tokens;
    std::string original_text;
};

} // namespace mira

