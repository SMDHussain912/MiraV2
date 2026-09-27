#include "core/text_utils.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace mira
{

namespace
{

bool is_space(unsigned char character)
{
    return std::isspace(character) != 0;
}

bool is_punctuation(unsigned char character)
{
    return std::ispunct(character) != 0;
}

char to_lower(unsigned char character)
{
    return static_cast<char>(std::tolower(character));
}

bool same_character_ignoring_case(char left, char right)
{
    return to_lower(static_cast<unsigned char>(left)) ==
           to_lower(static_cast<unsigned char>(right));
}

} // namespace

std::string normalize_for_matching(const std::string& text)
{
    std::string result;

    for (unsigned char character : text)
    {
        if (is_punctuation(character))
        {
            result += ' ';
        }
        else
        {
            result += to_lower(character);
        }
    }

    return result;
}

std::string trim(const std::string& text)
{
    std::string value = text;

    value.erase(
        value.begin(),
        std::find_if(
            value.begin(),
            value.end(),
            [](unsigned char character)
            {
                return !is_space(character);
            }
        )
    );

    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            [](unsigned char character)
            {
                return !is_space(character);
            }
        ).base(),
        value.end()
    );

    return value;
}

std::vector<std::string> split_whitespace(const std::string& text)
{
    std::vector<std::string> tokens;

    std::istringstream stream(text);

    std::string token;

    while (stream >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

bool equals_case_insensitive(
    const std::string& left,
    const std::string& right)
{
    if (left.size() != right.size())
    {
        return false;
    }

    return std::equal(
        left.begin(),
        left.end(),
        right.begin(),
        same_character_ignoring_case
    );
}

bool strip_prefix_case_insensitive(
    std::string& text,
    const std::string& prefix)
{
    if (text.size() < prefix.size())
    {
        return false;
    }

    const bool matches = std::equal(
        prefix.begin(),
        prefix.end(),
        text.begin(),
        same_character_ignoring_case
    );

    if (!matches)
    {
        return false;
    }

    text.erase(0, prefix.size());

    return true;
}

} // namespace mira
