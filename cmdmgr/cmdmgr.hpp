#pragma once

#include <string>
#include <vector>

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

private:
    std::string normalize(const std::string& text);
    std::vector<std::string> tokenize(const std::string& text);
};
