#include "core/intent.hpp"

namespace mira
{

namespace
{

struct IntentName
{
    Intent intent;
    const char* name;
};

// Order matters: it defines the option indices returned to the TAMEV model and
// must stay identical to the OPTIONS list used during training.
const IntentName kIntentNames[] = {
    { Intent::OpenApplication, "open_application" },
    { Intent::SearchWeb,       "search_web"       },
    { Intent::ReadScreen,      "read_screen"      },
    { Intent::TypeText,        "type_text"        },
    { Intent::FileOperation,   "file_operation"   },
    { Intent::SystemControl,   "system_control"   },
    { Intent::Conversation,    "conversation"     },
};

} // namespace

std::string to_string(Intent intent)
{
    for (const IntentName& entry : kIntentNames)
    {
        if (entry.intent == intent)
        {
            return entry.name;
        }
    }

    return "unknown";
}

Intent intent_from_string(const std::string& value)
{
    for (const IntentName& entry : kIntentNames)
    {
        if (value == entry.name)
        {
            return entry.intent;
        }
    }

    return Intent::Unknown;
}

const std::vector<Intent>& intent_catalog()
{
    static const std::vector<Intent> catalog = {
        Intent::OpenApplication,
        Intent::SearchWeb,
        Intent::ReadScreen,
        Intent::TypeText,
        Intent::FileOperation,
        Intent::SystemControl,
        Intent::Conversation,
    };

    return catalog;
}

int intent_to_option_index(Intent intent)
{
    const std::vector<Intent>& catalog = intent_catalog();

    for (std::size_t index = 0; index < catalog.size(); ++index)
    {
        if (catalog[index] == intent)
        {
            return static_cast<int>(index);
        }
    }

    return -1;
}

Intent intent_from_option_index(int index)
{
    const std::vector<Intent>& catalog = intent_catalog();

    if (index < 0 || static_cast<std::size_t>(index) >= catalog.size())
    {
        return Intent::Unknown;
    }

    return catalog[static_cast<std::size_t>(index)];
}

} // namespace mira
