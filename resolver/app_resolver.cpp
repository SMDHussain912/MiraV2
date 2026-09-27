#include "resolver/app_resolver.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

#include "core/text_utils.hpp"

namespace mira
{

namespace
{

static std::string to_lower_copy(const std::string& str)
{
    std::string s = str;
    for (char& c : s)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// Normalization helper for comparison only.
// Lowercases, strips punctuation, and collapses multiple whitespace characters.
// The original names/aliases/targets are never modified in storage or output.
std::string normalize_for_comparison(const std::string& input)
{
    std::string s = to_lower_copy(trim(input));
    std::string result;
    result.reserve(s.size());

    bool last_was_space = false;
    for (char c : s)
    {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
        {
            result.push_back(c);
            last_was_space = false;
        }
        else
        {
            if (!last_was_space && !result.empty())
            {
                result.push_back(' ');
                last_was_space = true;
            }
        }
    }

    return trim(result);
}

// Check if query is a prefix or word-boundary prefix in target.
bool is_bounded_word_match(const std::string& target_norm, const std::string& query_norm)
{
    if (query_norm.empty() || target_norm.empty())
    {
        return false;
    }

    if (query_norm.length() < 3)
    {
        return false;
    }

    if (target_norm == query_norm)
    {
        return true;
    }

    // Target starts with query as a prefix
    if (target_norm.rfind(query_norm, 0) == 0)
    {
        return true;
    }

    const std::string word_boundary = " " + query_norm;
    const std::size_t pos = target_norm.find(word_boundary);
    if (pos != std::string::npos)
    {
        const std::size_t end_pos = pos + word_boundary.length();
        if (end_pos == target_norm.length() || target_norm[end_pos] == ' ')
        {
            return true;
        }
    }

    return false;
}

} // namespace

AppResolver::AppResolver(ApplicationConfig config)
    : config_(std::move(config))
{
}

Result AppResolver::resolve(const TokenResult& token, LaunchSpec& out_spec) const
{
    if (!token.valid || token.status != TokenStatus::Success || token.type != TokenType::TARGET)
    {
        return Result::rejected(
            "invalid token for application resolution",
            "Token status: " + to_string(token.status) + ", type: " + to_string(token.type)
        );
    }

    return resolve(token.value, out_spec);
}

Result AppResolver::resolve(const std::string& app_reference, LaunchSpec& out_spec) const
{
    const std::string target = trim(app_reference);
    if (target.empty())
    {
        return Result::rejected("empty application target");
    }

    // Tier 1: Exact Match (Case-sensitive check against name/aliases)
    std::vector<const ApplicationEntry*> exact_matches;
    for (const auto& entry : config_.applications())
    {
        if (entry.name == target)
        {
            exact_matches.push_back(&entry);
            continue;
        }
        for (const auto& alias : entry.aliases)
        {
            if (alias == target)
            {
                exact_matches.push_back(&entry);
                break;
            }
        }
    }

    if (exact_matches.size() == 1)
    {
        out_spec = exact_matches[0]->launch;
        return Result::ok();
    }
    if (exact_matches.size() > 1)
    {
        std::string detail = "Candidates: ";
        for (std::size_t i = 0; i < exact_matches.size(); ++i)
        {
            if (i > 0) detail += ", ";
            detail += exact_matches[i]->name;
        }
        return Result::rejected("ambiguous application reference", detail);
    }

    // Tier 2: Normalized Match (Case-insensitive, punctuation/spacing normalized)
    const std::string target_norm = normalize_for_comparison(target);
    std::vector<const ApplicationEntry*> norm_matches;

    for (const auto& entry : config_.applications())
    {
        if (normalize_for_comparison(entry.name) == target_norm)
        {
            norm_matches.push_back(&entry);
            continue;
        }
        for (const auto& alias : entry.aliases)
        {
            if (normalize_for_comparison(alias) == target_norm)
            {
                norm_matches.push_back(&entry);
                break;
            }
        }
    }

    if (norm_matches.size() == 1)
    {
        out_spec = norm_matches[0]->launch;
        return Result::ok();
    }
    if (norm_matches.size() > 1)
    {
        std::string detail = "Candidates: ";
        for (std::size_t i = 0; i < norm_matches.size(); ++i)
        {
            if (i > 0) detail += ", ";
            detail += norm_matches[i]->name;
        }
        return Result::rejected("ambiguous application reference", detail);
    }

    // Tier 3: Bounded Word Boundary / Prefix Match (>= 3 chars)
    std::vector<const ApplicationEntry*> bounded_matches;
    for (const auto& entry : config_.applications())
    {
        if (is_bounded_word_match(normalize_for_comparison(entry.name), target_norm))
        {
            bounded_matches.push_back(&entry);
            continue;
        }
        for (const auto& alias : entry.aliases)
        {
            if (is_bounded_word_match(normalize_for_comparison(alias), target_norm))
            {
                bounded_matches.push_back(&entry);
                break;
            }
        }
    }

    if (bounded_matches.size() == 1)
    {
        out_spec = bounded_matches[0]->launch;
        return Result::ok();
    }
    if (bounded_matches.size() > 1)
    {
        std::string detail = "Candidates: ";
        for (std::size_t i = 0; i < bounded_matches.size(); ++i)
        {
            if (i > 0) detail += ", ";
            detail += bounded_matches[i]->name;
        }
        return Result::rejected("ambiguous application reference", detail);
    }

    // Tier 4: Not Configured
    return Result::rejected(
        "not configured",
        "No matching application for '" + target + "'"
    );
}

} // namespace mira
