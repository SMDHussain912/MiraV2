#include "core/action.hpp"

namespace mira
{

std::string to_string(TokenType type)
{
    switch (type)
    {
        case TokenType::TARGET:
            return "target";
        case TokenType::QUERY:
            return "query";
        case TokenType::COMMAND:
            return "command";
        case TokenType::TEXT:
            return "text";
        case TokenType::PATH:
            return "path";
        case TokenType::UNKNOWN:
        default:
            return "unknown";
    }
}

std::string to_string(TokenStatus status)
{
    switch (status)
    {
        case TokenStatus::Success:
            return "success";
        case TokenStatus::EmptyInput:
            return "empty_input";
        case TokenStatus::MissingTarget:
            return "missing_target";
        case TokenStatus::UnsupportedIntent:
            return "unsupported_intent";
        default:
            return "unknown";
    }
}

} // namespace mira
