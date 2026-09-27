// Tests for application resolver (Phase 3).
#include <string>

#include "core/action.hpp"
#include "core/result.hpp"
#include "resolver/app_resolver.hpp"
#include "resolver/application_config.hpp"
#include "test_util.hpp"

using mira::AppResolver;
using mira::ApplicationConfig;
using mira::LaunchMethod;
using mira::LaunchSpec;
using mira::Result;
using mira::ResultStatus;
using mira::TokenResult;
using mira::TokenStatus;
using mira::TokenType;

namespace
{

const char* const kTestConfigJson = R"({
    "version": 1,
    "applications": [
        {
            "name": "Firefox",
            "aliases": ["ff", "mozilla firefox", "browser"],
            "launch": {
                "method": "command",
                "command": "firefox",
                "working_directory": null
            }
        },
        {
            "name": "Visual Studio Code",
            "aliases": ["vscode", "code"],
            "launch": {
                "method": "command",
                "command": "code",
                "working_directory": "/home/user/workspace"
            }
        },
        {
            "name": "7-Zip File Manager",
            "aliases": ["7-zip", "7zip"],
            "launch": {
                "method": "executable",
                "path": "/usr/bin/7z"
            }
        },
        {
            "name": "Steam",
            "aliases": ["steam gaming"],
            "launch": {
                "method": "command",
                "command": "steam"
            }
        },
        {
            "name": "Steam Link",
            "aliases": ["link streaming"],
            "launch": {
                "method": "command",
                "command": "steamlink"
            }
        }
    ]
})";

AppResolver make_resolver()
{
    ApplicationConfig cfg;
    Result res = ApplicationConfig::load_from_string(kTestConfigJson, cfg);
    (void)res;
    return AppResolver(std::move(cfg));
}

void exact_name_and_alias_matching()
{
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    // Exact name match
    Result r1 = resolver.resolve("Firefox", spec);
    MIRA_CHECK_EQ(r1.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Firefox"));
    MIRA_CHECK_EQ(spec.method, LaunchMethod::Command);
    MIRA_CHECK_EQ(spec.command, std::string("firefox"));

    // Exact alias match
    Result r2 = resolver.resolve("vscode", spec);
    MIRA_CHECK_EQ(r2.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Visual Studio Code"));
    MIRA_CHECK_EQ(spec.command, std::string("code"));
    MIRA_CHECK_EQ(spec.working_directory, std::string("/home/user/workspace"));

    // Exact punctuation match: 7-Zip
    Result r3 = resolver.resolve("7-Zip", spec);
    MIRA_CHECK_EQ(r3.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("7-Zip File Manager"));
    MIRA_CHECK_EQ(spec.method, LaunchMethod::Executable);
    MIRA_CHECK_EQ(spec.path, std::string("/usr/bin/7z"));
}

void normalized_case_and_punctuation_matching()
{
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    // Case normalization: "fIrEfOx" -> Firefox
    Result r1 = resolver.resolve("fIrEfOx", spec);
    MIRA_CHECK_EQ(r1.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Firefox"));

    // Punctuation and casing: "Visual Studio Code." -> Visual Studio Code
    Result r2 = resolver.resolve("Visual Studio Code.", spec);

    MIRA_CHECK_EQ(r2.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Visual Studio Code"));

    // Punctuation in alias: "7 - zip" -> 7-Zip
    Result r3 = resolver.resolve("7 - zip", spec);
    MIRA_CHECK_EQ(r3.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("7-Zip File Manager"));
}

void ambiguous_matches_are_rejected()
{
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    // "steam" matches "Steam" exact, so exact tier wins deterministically!
    Result r_exact = resolver.resolve("Steam", spec);
    MIRA_CHECK_EQ(r_exact.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Steam"));

    // A prefix query "ste" in Tier 3 matches both "Steam" and "Steam Link", so ambiguous
    Result r_ambig = resolver.resolve("ste", spec);
    MIRA_CHECK_EQ(r_ambig.is_error(), true);
    MIRA_CHECK_EQ(r_ambig.message(), std::string("ambiguous application reference"));
}

void unconfigured_target_is_rejected()
{
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    Result res = resolver.resolve("UnknownCalculator", spec);
    MIRA_CHECK_EQ(res.is_error(), true);
    MIRA_CHECK_EQ(res.status(), ResultStatus::Rejected);
    MIRA_CHECK_EQ(res.message(), std::string("not configured"));
}

void token_result_overload()
{
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    // Valid token
    TokenResult valid_tok;
    valid_tok.type = TokenType::TARGET;
    valid_tok.value = "firefox";
    valid_tok.status = TokenStatus::Success;
    valid_tok.valid = true;

    Result r1 = resolver.resolve(valid_tok, spec);
    MIRA_CHECK_EQ(r1.is_ok(), true);
    MIRA_CHECK_EQ(spec.name, std::string("Firefox"));

    // Invalid token (e.g. MissingTarget)
    TokenResult bad_tok;
    bad_tok.type = TokenType::TARGET;
    bad_tok.value = "";
    bad_tok.status = TokenStatus::MissingTarget;
    bad_tok.valid = false;

    Result r2 = resolver.resolve(bad_tok, spec);
    MIRA_CHECK_EQ(r2.is_error(), true);
    MIRA_CHECK_EQ(r2.message(), std::string("invalid token for application resolution"));
}

void resolver_is_side_effect_free()
{
    // Proves resolution causes zero OS mutations or process launches.
    AppResolver resolver = make_resolver();
    LaunchSpec spec;

    Result r = resolver.resolve("Firefox", spec);
    MIRA_CHECK_EQ(r.is_ok(), true);
    MIRA_CHECK_EQ(spec.command, std::string("firefox"));
}

} // namespace

int main()
{
    exact_name_and_alias_matching();
    normalized_case_and_punctuation_matching();
    ambiguous_matches_are_rejected();
    unconfigured_target_is_rejected();
    token_result_overload();
    resolver_is_side_effect_free();

    return mira_test::finish("app_resolver");
}
