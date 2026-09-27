#pragma once

// LEGACY - scheduled for removal.
//
// CommandManager predates the intent pipeline. It is kept only so that the
// current runtime (main.cpp) keeps working while the new stages are built; it is
// retired in ROADMAP Phase 8, once TAMEV -> tokenizer -> executor handles a
// request end to end.
//
// Important: Command::action is the raw first word of the utterance. It is NOT a
// mira::Intent and must not be treated as one. Deciding the intent from the first
// word is precisely the behaviour this class is being replaced to remove, and
// nothing new may depend on its output.
//
// The class is intentionally not marked [[deprecated]]: that attribute would emit
// warnings for the current, still-supported caller in main.cpp (the build is kept
// warning-clean), so the deprecation is expressed here in text instead.

#include <string>

struct Command
{
    std::string action;
    std::string target;
    bool valid;
};

class CommandManager
{
public:
    Command process(const std::string& text);
};

