// Unit tests for the canonical intent vocabulary (core/intent.*).
//
// Two things here are contracts with the TAMEV model rather than style choices:
//   * the string form must match the option names used during training;
//   * the catalog order must match the position the model was trained to predict,
//     because the model returns a distribution over a positional option list.

#include "core/intent.hpp"
#include "test_util.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace
{

using mira::Intent;

void canonical_string_forms()
{
    MIRA_CHECK_EQ(mira::to_string(Intent::OpenApplication), std::string("open_application"));
    MIRA_CHECK_EQ(mira::to_string(Intent::SearchWeb), std::string("search_web"));
    MIRA_CHECK_EQ(mira::to_string(Intent::ReadScreen), std::string("read_screen"));
    MIRA_CHECK_EQ(mira::to_string(Intent::TypeText), std::string("type_text"));
    MIRA_CHECK_EQ(mira::to_string(Intent::FileOperation), std::string("file_operation"));
    MIRA_CHECK_EQ(mira::to_string(Intent::SystemControl), std::string("system_control"));
    MIRA_CHECK_EQ(mira::to_string(Intent::Conversation), std::string("conversation"));
    MIRA_CHECK_EQ(mira::to_string(Intent::Unknown), std::string("unknown"));
}

void every_intent_round_trips()
{
    const std::vector<Intent> all_intents = {
        Intent::OpenApplication,
        Intent::SearchWeb,
        Intent::ReadScreen,
        Intent::TypeText,
        Intent::FileOperation,
        Intent::SystemControl,
        Intent::Conversation,
        Intent::Unknown,
    };

    for (Intent intent : all_intents)
    {
        MIRA_CHECK_EQ(mira::intent_from_string(mira::to_string(intent)), intent);
    }
}

void unrecognised_strings_become_unknown()
{
    // The uppercase vocabulary used before Phase 1 no longer exists: intent
    // strings are lowercase snake_case only.
    MIRA_CHECK_EQ(mira::intent_from_string("OPEN_APPLICATION"), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_string(""), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_string("open"), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_string("Open_Application"), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_string("open_application "), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_string("not_an_intent"), Intent::Unknown);
}

void catalog_matches_the_training_option_order()
{
    const std::vector<Intent>& catalog = mira::intent_catalog();

    MIRA_CHECK_EQ(catalog.size(), std::size_t{7});

    MIRA_CHECK_EQ(catalog[0], Intent::OpenApplication);
    MIRA_CHECK_EQ(catalog[1], Intent::SearchWeb);
    MIRA_CHECK_EQ(catalog[2], Intent::ReadScreen);
    MIRA_CHECK_EQ(catalog[3], Intent::TypeText);
    MIRA_CHECK_EQ(catalog[4], Intent::FileOperation);
    MIRA_CHECK_EQ(catalog[5], Intent::SystemControl);
    MIRA_CHECK_EQ(catalog[6], Intent::Conversation);

    // Unknown is an abstention, not an option the model chooses between.
    MIRA_CHECK_EQ(mira::intent_to_option_index(Intent::Unknown), -1);
}

void option_indices_round_trip()
{
    const std::vector<Intent>& catalog = mira::intent_catalog();

    for (std::size_t index = 0; index < catalog.size(); ++index)
    {
        MIRA_CHECK_EQ(
            mira::intent_to_option_index(catalog[index]),
            static_cast<int>(index)
        );

        MIRA_CHECK_EQ(
            mira::intent_from_option_index(static_cast<int>(index)),
            catalog[index]
        );
    }

    MIRA_CHECK_EQ(mira::intent_from_option_index(-1), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_option_index(7), Intent::Unknown);
    MIRA_CHECK_EQ(mira::intent_from_option_index(1000), Intent::Unknown);
}

} // namespace

int main()
{
    canonical_string_forms();
    every_intent_round_trips();
    unrecognised_strings_become_unknown();
    catalog_matches_the_training_option_order();
    option_indices_round_trip();

    return mira_test::finish("intent");
}
