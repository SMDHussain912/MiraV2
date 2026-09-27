// Tests for application configuration loading and schema validation (Phase 3).
#include "core/paths.hpp"
#include "core/result.hpp"
#include "resolver/application_config.hpp"
#include "test_util.hpp"

#include <string>

using mira::ApplicationConfig;
using mira::LaunchMethod;
using mira::Result;
using mira::ResultStatus;

namespace
{

void valid_json_all_five_methods()
{
    const std::string valid_json = R"({
        "version": 1,
        "applications": [
            {
                "name": "Firefox",
                "aliases": ["ff", "web browser"],
                "launch": {
                    "method": "command",
                    "command": "firefox --new-tab",
                    "working_directory": "/home/user"
                }
            },
            {
                "name": "Custom Executable",
                "launch": {
                    "method": "executable",
                    "path": "/usr/local/bin/custom_exec",
                    "working_directory": null
                }
            },
            {
                "name": "Backup Script",
                "aliases": ["backup"],
                "launch": {
                    "method": "script",
                    "path": "/opt/scripts/backup.sh"
                }
            },
            {
                "name": "Tool AppImage",
                "launch": {
                    "method": "appimage",
                    "path": "/opt/tools/tool.AppImage"
                }
            },
            {
                "name": "GIMP Image Editor",
                "aliases": ["gimp"],
                "launch": {
                    "method": "desktop",
                    "desktop_file": "gimp.desktop"
                }
            }
        ]
    })";

    ApplicationConfig config;
    Result res = ApplicationConfig::load_from_string(valid_json, config);
    MIRA_CHECK_EQ(res.is_ok(), true);
    MIRA_CHECK_EQ(config.version(), 1);
    MIRA_CHECK_EQ(config.size(), 5u);

    // Entry 0: Command
    MIRA_CHECK_EQ(config.applications()[0].name, std::string("Firefox"));
    MIRA_CHECK_EQ(config.applications()[0].aliases.size(), 2u);
    MIRA_CHECK_EQ(config.applications()[0].launch.method, LaunchMethod::Command);
    MIRA_CHECK_EQ(config.applications()[0].launch.command, std::string("firefox --new-tab"));
    MIRA_CHECK_EQ(config.applications()[0].launch.working_directory, std::string("/home/user"));

    // Entry 1: Executable
    MIRA_CHECK_EQ(config.applications()[1].name, std::string("Custom Executable"));
    MIRA_CHECK_EQ(config.applications()[1].launch.method, LaunchMethod::Executable);
    MIRA_CHECK_EQ(config.applications()[1].launch.path, std::string("/usr/local/bin/custom_exec"));

    // Entry 2: Script
    MIRA_CHECK_EQ(config.applications()[2].name, std::string("Backup Script"));
    MIRA_CHECK_EQ(config.applications()[2].launch.method, LaunchMethod::Script);
    MIRA_CHECK_EQ(config.applications()[2].launch.path, std::string("/opt/scripts/backup.sh"));

    // Entry 3: AppImage
    MIRA_CHECK_EQ(config.applications()[3].name, std::string("Tool AppImage"));
    MIRA_CHECK_EQ(config.applications()[3].launch.method, LaunchMethod::AppImage);
    MIRA_CHECK_EQ(config.applications()[3].launch.path, std::string("/opt/tools/tool.AppImage"));

    // Entry 4: Desktop
    MIRA_CHECK_EQ(config.applications()[4].name, std::string("GIMP Image Editor"));
    MIRA_CHECK_EQ(config.applications()[4].launch.method, LaunchMethod::Desktop);
    MIRA_CHECK_EQ(config.applications()[4].launch.desktop_file, std::string("gimp.desktop"));
}

void malformed_json_syntax_rejected()
{
    ApplicationConfig config;
    Result res = ApplicationConfig::load_from_string("{ not valid json }", config);
    MIRA_CHECK_EQ(res.is_error(), true);
    MIRA_CHECK_EQ(res.status(), ResultStatus::Rejected);
}

void wrong_version_rejected()
{
    const std::string json = R"({"version": 2, "applications": []})";
    ApplicationConfig config;
    Result res = ApplicationConfig::load_from_string(json, config);
    MIRA_CHECK_EQ(res.is_error(), true);
    MIRA_CHECK_EQ(res.message(), std::string("unsupported configuration version"));
}

void missing_required_fields_rejected()
{
    ApplicationConfig config;

    // Missing 'applications'
    Result res1 = ApplicationConfig::load_from_string(R"({"version": 1})", config);
    MIRA_CHECK_EQ(res1.is_error(), true);

    // Missing application 'name'
    const std::string no_name = R"({
        "version": 1,
        "applications": [{"launch": {"method": "command", "command": "ls"}}]
    })";
    Result res2 = ApplicationConfig::load_from_string(no_name, config);
    MIRA_CHECK_EQ(res2.is_error(), true);

    // Missing launch object
    const std::string no_launch = R"({
        "version": 1,
        "applications": [{"name": "app"}]
    })";
    Result res3 = ApplicationConfig::load_from_string(no_launch, config);
    MIRA_CHECK_EQ(res3.is_error(), true);

    // Method command missing 'command'
    const std::string no_cmd = R"({
        "version": 1,
        "applications": [{"name": "app", "launch": {"method": "command"}}]
    })";
    Result res4 = ApplicationConfig::load_from_string(no_cmd, config);
    MIRA_CHECK_EQ(res4.is_error(), true);

    // Method executable missing 'path'
    const std::string no_path = R"({
        "version": 1,
        "applications": [{"name": "app", "launch": {"method": "executable"}}]
    })";
    Result res5 = ApplicationConfig::load_from_string(no_path, config);
    MIRA_CHECK_EQ(res5.is_error(), true);

    // Method desktop missing 'desktop_file'
    const std::string no_desktop = R"({
        "version": 1,
        "applications": [{"name": "app", "launch": {"method": "desktop"}}]
    })";
    Result res6 = ApplicationConfig::load_from_string(no_desktop, config);
    MIRA_CHECK_EQ(res6.is_error(), true);

    // Unsupported method
    const std::string unknown_method = R"({
        "version": 1,
        "applications": [{"name": "app", "launch": {"method": "teleport"}}]
    })";
    Result res7 = ApplicationConfig::load_from_string(unknown_method, config);
    MIRA_CHECK_EQ(res7.is_error(), true);
}

void duplicate_names_and_aliases_rejected()
{
    ApplicationConfig config;

    // Duplicate name (case-insensitive check)
    const std::string dup_name = R"({
        "version": 1,
        "applications": [
            {"name": "Firefox", "launch": {"method": "command", "command": "firefox"}},
            {"name": "firefox", "launch": {"method": "command", "command": "firefox2"}}
        ]
    })";
    Result res1 = ApplicationConfig::load_from_string(dup_name, config);
    MIRA_CHECK_EQ(res1.is_error(), true);
    MIRA_CHECK_EQ(res1.message(), std::string("duplicate application name"));

    // Duplicate alias across applications
    const std::string dup_alias = R"({
        "version": 1,
        "applications": [
            {"name": "App1", "aliases": ["shared"], "launch": {"method": "command", "command": "a1"}},
            {"name": "App2", "aliases": ["shared"], "launch": {"method": "command", "command": "a2"}}
        ]
    })";
    Result res2 = ApplicationConfig::load_from_string(dup_alias, config);
    MIRA_CHECK_EQ(res2.is_error(), true);
    MIRA_CHECK_EQ(res2.message(), std::string("duplicate alias"));
}

void load_committed_default_file()
{
    const std::string default_path = mira::paths::resolve_data_path("config/apps.json");
    MIRA_CHECK_EQ(default_path.empty(), false);

    ApplicationConfig config;
    Result res = ApplicationConfig::load_from_file(default_path, config);
    MIRA_CHECK_EQ(res.is_ok(), true);
    MIRA_CHECK_EQ(config.version(), 1);
    MIRA_CHECK_EQ(config.empty(), false);
}

} // namespace

int main()
{
    valid_json_all_five_methods();
    malformed_json_syntax_rejected();
    wrong_version_rejected();
    missing_required_fields_rejected();
    duplicate_names_and_aliases_rejected();
    load_committed_default_file();

    return mira_test::finish("app_config");
}
