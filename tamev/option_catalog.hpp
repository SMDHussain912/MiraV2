#pragma once

#include <string>
#include <vector>

#include "core/intent.hpp"

namespace mira
{

// Canonical TAMEV inference contract (ROADMAP Phase 6).
//
// Exactly one standard context format and padding scheme is used by both the
// Python reference (models/training/train.py) and this C++ runtime, resolving
// the historical tester.py vs train.py divergence:
//
//   context = QUESTION + "\nUser request: " + <utterance>   (ctx_len = 128)
//   options = full 7-entry catalog below, in order           (opt_len = 64)
//
// Order is a hard contract: the model returns a distribution over positional
// indices, so this list must stay identical to the OPTIONS list used during
// training and to core::intent_catalog().
constexpr const char* TAMEV_QUESTION = "Which action should Mira perform?";

constexpr std::size_t TAMEV_CONTEXT_LENGTH = 128;
constexpr std::size_t TAMEV_OPTION_LENGTH = 64;

// Builds the model input context for one user utterance.
inline std::string tamev_context_text(const std::string& utterance)
{
    return std::string(TAMEV_QUESTION) + "\nUser request: " + utterance;
}

// Returns the candidate options list in exact positional training order.
// This is a hard contract with the TAMEV dual-encoder model.
const std::vector<std::string>& tamev_option_strings();

// Map positional index from TAMEV model prediction to core::Intent.
inline Intent tamev_index_to_intent(size_t index)
{
    return intent_from_option_index(static_cast<int>(index));
}

// Map core::Intent to positional index in TAMEV options list (or -1 if not found).
inline int tamev_intent_to_index(Intent intent)
{
    return intent_to_option_index(intent);
}

} // namespace mira


