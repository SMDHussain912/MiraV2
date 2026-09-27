// Characterisation test for the legacy command splitter (cmdmgr/).
//
// This test does not claim that the behaviour below is desirable. It pins the
// behaviour that exists today so that Phase 1 (shared text utilities and core
// interfaces) and Phase 8 (retiring CommandManager) cannot change it by
// accident. When the splitter is eventually removed, these expectations move
// to the tokenizer tests or are deleted deliberately.
//
// Documented limitations of the current implementation, encoded below as
// expectations:
//   * the "action" is simply the first word;
//   * any two-word sentence is therefore a "valid" command, including
//     questions such as "What is on my screen?";
//   * punctuation is replaced by spaces, so "fire-fox" becomes "fire fox".

#include "cmdmgr/cmdmgr.hpp"
#include "test_util.hpp"

#include <string>

namespace
{

void two_word_command_is_split_into_action_and_target()
{
    CommandManager manager;

    const Command command = manager.process("Open Firefox");

    MIRA_CHECK_EQ(command.action, std::string("open"));
    MIRA_CHECK_EQ(command.target, std::string("firefox"));
    MIRA_CHECK_EQ(command.valid, true);
}

void text_is_case_folded()
{
    CommandManager manager;

    const Command command = manager.process("LAUNCH Blender");

    MIRA_CHECK_EQ(command.action, std::string("launch"));
    MIRA_CHECK_EQ(command.target, std::string("blender"));
    MIRA_CHECK_EQ(command.valid, true);
}

void repeated_whitespace_collapses()
{
    CommandManager manager;

    const Command command = manager.process("open     firefox");

    MIRA_CHECK_EQ(command.action, std::string("open"));
    MIRA_CHECK_EQ(command.target, std::string("firefox"));
    MIRA_CHECK_EQ(command.valid, true);
}

void trailing_punctuation_is_dropped()
{
    CommandManager manager;

    const Command command = manager.process("Open Firefox!");

    MIRA_CHECK_EQ(command.action, std::string("open"));
    MIRA_CHECK_EQ(command.target, std::string("firefox"));
    MIRA_CHECK_EQ(command.valid, true);
}

void punctuation_inside_a_word_becomes_a_separator()
{
    CommandManager manager;

    const Command command = manager.process("open fire-fox");

    MIRA_CHECK_EQ(command.target, std::string("fire fox"));
    MIRA_CHECK_EQ(command.valid, true);
}

void multi_word_targets_are_joined()
{
    CommandManager manager;

    const Command command = manager.process("open unity editor");

    MIRA_CHECK_EQ(command.action, std::string("open"));
    MIRA_CHECK_EQ(command.target, std::string("unity editor"));
    MIRA_CHECK_EQ(command.valid, true);
}

void a_command_without_a_target_is_invalid()
{
    CommandManager manager;

    const Command command = manager.process("Open");

    MIRA_CHECK_EQ(command.action, std::string("open"));
    MIRA_CHECK_EQ(command.target, std::string(""));
    MIRA_CHECK_EQ(command.valid, false);
}

void empty_input_is_invalid()
{
    CommandManager manager;

    const Command command = manager.process("");

    MIRA_CHECK_EQ(command.action, std::string(""));
    MIRA_CHECK_EQ(command.target, std::string(""));
    MIRA_CHECK_EQ(command.valid, false);
}

void whitespace_only_input_is_invalid()
{
    CommandManager manager;

    const Command command = manager.process("   ");

    MIRA_CHECK_EQ(command.valid, false);
}

void a_question_is_treated_as_a_valid_command()
{
    // Documents the known defect that motivates intent routing: the splitter
    // has no notion of intent, so it accepts a question as a command.
    CommandManager manager;

    const Command command = manager.process("What is on my screen?");

    MIRA_CHECK_EQ(command.action, std::string("what"));
    MIRA_CHECK_EQ(command.target, std::string("is on my screen"));
    MIRA_CHECK_EQ(command.valid, true);
}

} // namespace

int main()
{
    two_word_command_is_split_into_action_and_target();
    text_is_case_folded();
    repeated_whitespace_collapses();
    trailing_punctuation_is_dropped();
    punctuation_inside_a_word_becomes_a_separator();
    multi_word_targets_are_joined();
    a_command_without_a_target_is_invalid();
    empty_input_is_invalid();
    whitespace_only_input_is_invalid();
    a_question_is_treated_as_a_valid_command();

    return mira_test::finish("cmdmgr_baseline");
}
