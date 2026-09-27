#pragma once

#include <string>
#include <vector>

namespace mira
{

// The canonical intent vocabulary of MiraV2.
//
// This is the single definition of "what Mira was asked to do". Every component
// that needs to talk about intent (TAMEV, the tokenizer, the executor) uses this
// enum; no component may define its own intent vocabulary or string table.
//
// The string form is lowercase snake_case and matches the candidate option names
// used when the TAMEV model is trained (models/training/train.py). That is a hard
// contract: the model returns a distribution over a positional option list, so
// the order of intent_catalog() must match the order used in training.
enum class Intent
{
    OpenApplication,
    SearchWeb,
    ReadScreen,
    TypeText,
    FileOperation,
    SystemControl,
    Conversation,

    // The model is not confident enough to choose an action, or the request was
    // not understood. Deliberately not part of the model's option list.
    Unknown
};

// Canonical string form, e.g. Intent::OpenApplication -> "open_application".
std::string to_string(Intent intent);

// Parses the canonical string form. Anything unrecognised (including the
// historical uppercase form "OPEN_APPLICATION") yields Intent::Unknown.
Intent intent_from_string(const std::string& value);

// The options offered to the TAMEV model, in the exact order used in training.
// Intent::Unknown is not part of this list.
const std::vector<Intent>& intent_catalog();

// Position of an intent in intent_catalog(), or -1 if it is not a model option.
int intent_to_option_index(Intent intent);

// Inverse of intent_to_option_index(); out-of-range indices yield Intent::Unknown.
Intent intent_from_option_index(int index);

} // namespace mira
