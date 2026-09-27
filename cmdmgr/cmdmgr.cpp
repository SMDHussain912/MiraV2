#include "cmdmgr.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

std::string CommandManager::normalize(const std::string& text)
{
    std::string result;

    for (unsigned char character : text)
    {
        if (std::ispunct(character))
        {
            result += ' ';
        }
        else
        {
            result += static_cast<char>(
                std::tolower(character)
            );
        }
    }

    return result;
}

std::vector<std::string>
CommandManager::tokenize(const std::string& text)
{
    std::vector<std::string> tokens;

    std::string normalized =
        normalize(text);

    std::istringstream stream(normalized);

    std::string token;

    while (stream >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

Command CommandManager::process(
    const std::string& text)
{
    Command command;

    command.action.clear();
    command.target.clear();
    command.valid = false;

    std::vector<std::string> tokens =
        tokenize(text);

    if (tokens.empty())
    {
        return command;
    }

    command.action = tokens[0];

    for (size_t i = 1; i < tokens.size(); ++i)
    {
        if (!command.target.empty())
        {
            command.target += ' ';
        }

        command.target += tokens[i];
    }

    if (!command.action.empty() &&
        !command.target.empty())
    {
        command.valid = true;
    }

    return command;
}
