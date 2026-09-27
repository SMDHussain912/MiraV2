#pragma once

#include <string>
#include <vector>

#include "core/result.hpp"
#include "resolver/launch_spec.hpp"

namespace mira
{

// A single application profile configured by the user.
struct ApplicationEntry
{
    std::string name;                          // Canonical human-readable name
    std::vector<std::string> aliases;          // Spoken/written aliases or nicknames
    LaunchSpec launch;                         // Concrete launch details
};

// Represents the parsed and validated user application configuration.
class ApplicationConfig
{
public:
    static constexpr int kCurrentVersion = 1;

    ApplicationConfig() = default;

    // Load and validate from a JSON file path.
    static Result load_from_file(const std::string& path, ApplicationConfig& out_config);

    // Load and validate from a raw JSON string (useful for testing and in-memory configs).
    static Result load_from_string(const std::string& json_str, ApplicationConfig& out_config);

    int version() const { return version_; }
    const std::vector<ApplicationEntry>& applications() const { return applications_; }

    bool empty() const { return applications_.empty(); }
    std::size_t size() const { return applications_.size(); }

private:
    int version_ = kCurrentVersion;
    std::vector<ApplicationEntry> applications_;
};

} // namespace mira
