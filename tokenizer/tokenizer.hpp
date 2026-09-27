#pragma once

#include <string>

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
    TokenType type;
    std::string value;
    bool valid;
};

class Tokenizer
{
public:
    TokenResult process(
        const std::string& text,
        const std::string& intent
    );

private:
    TokenResult extract_application(const std::string& text);
};
