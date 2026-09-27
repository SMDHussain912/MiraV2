#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>

TokenResult Tokenizer::process(
    const std::string& text,
    const std::string& intent
)
{
    TokenResult result;

    result.type = TokenType::UNKNOWN;
    result.value = "";
    result.valid = false;

    if (intent == "OPEN_APPLICATION")
    {
        return extract_application(text);
    }

    return result;
}

TokenResult Tokenizer::extract_application(const std::string& text)
{
    TokenResult result;

    result.type = TokenType::TARGET;
    result.value = "";
    result.valid = false;

    std::string value = text;

    // Remove leading and trailing spaces
    value.erase(
        value.begin(),
        std::find_if(
            value.begin(),
            value.end(),
            [](unsigned char ch)
            {
                return !std::isspace(ch);
            }
        )
    );

    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            [](unsigned char ch)
            {
                return !std::isspace(ch);
            }
        ).base(),
        value.end()
    );

    // Remove the command word
    const std::string commands[] = {
        "open ",
        "launch ",
        "start ",
        "run "
    };

    for (const std::string& command : commands)
    {
        if (value.size() >= command.size() &&
            std::equal(
                command.begin(),
                command.end(),
                value.begin(),
                [](char a, char b)
                {
                    return std::tolower(
                        static_cast<unsigned char>(a)
                    ) ==
                    std::tolower(
                        static_cast<unsigned char>(b)
                    );
                }
            ))
        {
            value.erase(0, command.size());
            break;
        }
    }

    result.value = value;
    result.valid = !value.empty();

    return result;
}
