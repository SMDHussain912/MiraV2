#pragma once

#include <string>

#include "core/action.hpp"
#include "core/intent.hpp"

namespace mira
{

// Intent-aware extraction.
//
// The tokenizer runs *after* TAMEV has decided the intent, and extracts only the
// value that intent needs. It never decides or guesses the intent: the intent is
// a required argument, so no caller can extract before classifying.
//
// Implemented: OpenApplication. Extraction for the remaining intents is added
// one intent at a time (ROADMAP Phase 8), each with its own tests.
class Tokenizer
{
public:
    TokenResult process(const std::string& text, Intent intent);

private:
    TokenResult extract_application(const std::string& text);
};

} // namespace mira

