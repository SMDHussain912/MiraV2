#include "tamev/option_catalog.hpp"

namespace mira
{

const std::vector<std::string>& tamev_option_strings()
{
    static const std::vector<std::string> options = []() {
        std::vector<std::string> strings;
        for (Intent intent : intent_catalog())
        {
            strings.push_back(to_string(intent));
        }
        return strings;
    }();
    return options;
}

} // namespace mira

