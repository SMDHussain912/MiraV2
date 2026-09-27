#pragma once

#include <string>

#include "core/action.hpp"
#include "core/result.hpp"
#include "resolver/application_config.hpp"
#include "resolver/launch_spec.hpp"

namespace mira
{

// Resolves application references (extracted by Tokenizer) into LaunchSpec definitions.
//
// AppResolver is strictly side-effect free:
//   - Does NOT launch applications
//   - Does NOT fork/exec or invoke system()
//   - Does NOT probe or kill processes
//   - Does NOT touch environment variables, D-Bus, or desktop portals
//   - Does NOT perform runtime OS execution
//
// Pipeline position:
//   TAMEV (Intent) -> Tokenizer (TokenResult) -> AppResolver (LaunchSpec) -> AppManager (Launch)
class AppResolver
{
public:
    explicit AppResolver(ApplicationConfig config);

    // Primary entry point accepting the Tokenizer's extraction output.
    Result resolve(const TokenResult& token, LaunchSpec& out_spec) const;

    // Direct string lookup overload (useful for programmatic lookups and unit tests).
    Result resolve(const std::string& app_reference, LaunchSpec& out_spec) const;

    const ApplicationConfig& config() const { return config_; }

private:
    ApplicationConfig config_;
};

} // namespace mira
