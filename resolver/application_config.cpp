#include "resolver/application_config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>

#include "core/text_utils.hpp"

namespace mira
{

using json = nlohmann::json;

static std::string to_lower_copy(const std::string& str)
{
    std::string s = str;
    for (char& c : s)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string to_string(LaunchMethod method)
{
    switch (method)
    {
        case LaunchMethod::Command:
            return "command";
        case LaunchMethod::Executable:
            return "executable";
        case LaunchMethod::Script:
            return "script";
        case LaunchMethod::AppImage:
            return "appimage";
        case LaunchMethod::Desktop:
            return "desktop";
        default:
            return "unknown";
    }
}

LaunchMethod launch_method_from_string(const std::string& str)
{
    const std::string lower = to_lower_copy(trim(str));
    if (lower == "command")
    {
        return LaunchMethod::Command;
    }
    if (lower == "executable")
    {
        return LaunchMethod::Executable;
    }
    if (lower == "script")
    {
        return LaunchMethod::Script;
    }
    if (lower == "appimage")
    {
        return LaunchMethod::AppImage;
    }
    if (lower == "desktop")
    {
        return LaunchMethod::Desktop;
    }
    return LaunchMethod::Command;
}

Result ApplicationConfig::load_from_file(const std::string& path, ApplicationConfig& out_config)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        return Result::rejected(
            "failed to open configuration file",
            "File path: " + path
        );
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_from_string(buffer.str(), out_config);
}

Result ApplicationConfig::load_from_string(const std::string& json_str, ApplicationConfig& out_config)
{
    json root;
    try
    {
        root = json::parse(json_str);
    }
    catch (const json::parse_error& e)
    {
        return Result::rejected("malformed configuration JSON", e.what());
    }

    if (!root.is_object())
    {
        return Result::rejected("invalid configuration format", "Root must be a JSON object");
    }

    if (!root.contains("version") || !root["version"].is_number_integer())
    {
        return Result::rejected("invalid configuration format", "Missing integer field 'version'");
    }

    const int ver = root["version"].get<int>();
    if (ver != kCurrentVersion)
    {
        return Result::rejected(
            "unsupported configuration version",
            "Expected version " + std::to_string(kCurrentVersion) + ", got " + std::to_string(ver)
        );
    }

    if (!root.contains("applications") || !root["applications"].is_array())
    {
        return Result::rejected("invalid configuration format", "Missing array field 'applications'");
    }

    ApplicationConfig parsed;
    parsed.version_ = ver;

    std::unordered_set<std::string> seen_names;
    std::unordered_set<std::string> seen_aliases;

    for (std::size_t i = 0; i < root["applications"].size(); ++i)
    {
        const auto& app_node = root["applications"][i];
        const std::string idx = "applications[" + std::to_string(i) + "]: ";

        if (!app_node.is_object())
        {
            return Result::rejected("invalid application entry", idx + "Must be a JSON object");
        }

        if (!app_node.contains("name") || !app_node["name"].is_string())
        {
            return Result::rejected("missing application name", idx + "Must contain string 'name'");
        }

        const std::string name = trim(app_node["name"].get<std::string>());
        if (name.empty())
        {
            return Result::rejected("empty application name", idx + "'name' cannot be blank");
        }

        const std::string norm_name = to_lower_copy(name);
        if (seen_names.count(norm_name) > 0)
        {
            return Result::rejected("duplicate application name", idx + "Name '" + name + "' already defined");
        }
        seen_names.insert(norm_name);

        ApplicationEntry entry;
        entry.name = name;

        if (app_node.contains("aliases"))
        {
            if (!app_node["aliases"].is_array())
            {
                return Result::rejected("invalid aliases field", idx + "'aliases' must be an array");
            }
            for (const auto& a_node : app_node["aliases"])
            {
                if (!a_node.is_string())
                {
                    return Result::rejected("invalid alias element", idx + "Each alias must be a string");
                }
                const std::string alias = trim(a_node.get<std::string>());
                if (!alias.empty())
                {
                    const std::string norm_a = to_lower_copy(alias);
                    if (seen_aliases.count(norm_a) > 0)
                    {
                        return Result::rejected("duplicate alias", idx + "Alias '" + alias + "' already assigned");
                    }
                    seen_aliases.insert(norm_a);
                    entry.aliases.push_back(alias);
                }
            }
        }

        if (!app_node.contains("launch") || !app_node["launch"].is_object())
        {
            return Result::rejected("missing launch configuration", idx + "Must contain 'launch' object");
        }

        const auto& launch_node = app_node["launch"];
        if (!launch_node.contains("method") || !launch_node["method"].is_string())
        {
            return Result::rejected("missing launch method", idx + "'launch' must contain string 'method'");
        }

        const std::string method_str = trim(launch_node["method"].get<std::string>());
        const std::string method_lower = to_lower_copy(method_str);

        if (method_lower != "command" && method_lower != "executable" &&
            method_lower != "script" && method_lower != "appimage" && method_lower != "desktop")
        {
            return Result::rejected("unsupported launch method", idx + "Unknown method '" + method_str + "'");
        }

        LaunchSpec spec;
        spec.name = name;
        spec.method = launch_method_from_string(method_lower);

        if (spec.method == LaunchMethod::Command)
        {
            if (!launch_node.contains("command") || !launch_node["command"].is_string() ||
                trim(launch_node["command"].get<std::string>()).empty())
            {
                return Result::rejected("missing launch command", idx + "Method 'command' requires 'command'");
            }
            spec.command = trim(launch_node["command"].get<std::string>());
        }
        else if (spec.method == LaunchMethod::Desktop)
        {
            if (!launch_node.contains("desktop_file") || !launch_node["desktop_file"].is_string() ||
                trim(launch_node["desktop_file"].get<std::string>()).empty())
            {
                return Result::rejected("missing desktop_file", idx + "Method 'desktop' requires 'desktop_file'");
            }
            spec.desktop_file = trim(launch_node["desktop_file"].get<std::string>());
        }
        else
        {
            if (!launch_node.contains("path") || !launch_node["path"].is_string() ||
                trim(launch_node["path"].get<std::string>()).empty())
            {
                return Result::rejected("missing launch path", idx + "Method '" + method_str + "' requires 'path'");
            }
            spec.path = trim(launch_node["path"].get<std::string>());
        }

        if (launch_node.contains("working_directory") && !launch_node["working_directory"].is_null())
        {
            if (!launch_node["working_directory"].is_string())
            {
                return Result::rejected("invalid working_directory", idx + "'working_directory' must be string or null");
            }
            spec.working_directory = trim(launch_node["working_directory"].get<std::string>());
        }

        entry.launch = spec;
        parsed.applications_.push_back(entry);
    }

    out_config = std::move(parsed);
    return Result::ok();
}

} // namespace mira
