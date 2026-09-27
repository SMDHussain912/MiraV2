// LEGACY implementation - see cmdmgr.hpp and ROADMAP Phase 8.
//
// Behaviour is unchanged from the original first-word splitter and is pinned by
// tests/test_cmdmgr_baseline.cpp. Text handling now comes from core/text_utils so
// that normalisation exists in exactly one place.

#include "cmdmgr/cmdmgr.hpp"

#include "core/text_utils.hpp"

#include <cstddef>
#include <vector>

Command CommandManager::process(const std::string& text)
{
    Command command;

    command.action.clear();
    command.target.clear();
    command.valid = false;

    const std::vector<std::string> tokens =
        mira::split_whitespace(mira::normalize_for_matching(text));

    if (tokens.empty())
    {
        return command;
    }

    command.action = tokens[0];

    for (std::size_t index = 1; index < tokens.size(); ++index)
    {
        if (!command.target.empty())
        {
            command.target += ' ';
        }

        command.target += tokens[index];
    }

    if (!command.action.empty() && !command.target.empty())
    {
        command.valid = true;
    }

    return command;
}

