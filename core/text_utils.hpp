#pragma once

#include <string>
#include <vector>

namespace mira
{

// Shared text handling for MiraV2.
//
// These helpers exist so that normalisation lives in exactly one place. The
// legacy CommandManager (removed in Phase 8) and the tokenizer previously
// carried their own copies; the tokenizer and pipeline now delegate here.
// Behaviour is pinned by tests/test_text_utils.cpp.

// Lowercases ASCII characters and replaces punctuation with spaces.
std::string normalize_for_matching(const std::string& text);

// Removes leading and trailing whitespace.
std::string trim(const std::string& text);

// Splits on runs of whitespace; leading and trailing whitespace is ignored.
std::vector<std::string> split_whitespace(const std::string& text);

// ASCII case-insensitive equality.
bool equals_case_insensitive(
    const std::string& left,
    const std::string& right);

// If `text` starts with `prefix` (case-insensitive), removes that prefix from
// `text` and returns true. Otherwise leaves `text` unchanged and returns false.
bool strip_prefix_case_insensitive(
    std::string& text,
    const std::string& prefix);

} // namespace mira
